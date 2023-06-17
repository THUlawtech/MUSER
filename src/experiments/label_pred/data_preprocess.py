import json
import urllib
import re
from urllib import request
from tqdm import tqdm

ah_pat = re.compile(r'[〔（(]\d\s*\d\s*\d\s*\d\s*[)）〕][^号]*号')
date_pat = re.compile(r'\d*年\d*月\d*日')
star_pat = re.compile(r'\**')
prefix_pat = re.compile(r'(一|原)审法院((认定(以下)*的*事实(如下)*：*)|((经*审理|另)*查明：))')
punt_pat = "[^A-Za-z\u4e00-\u9fa5]"

re.sub(punt_pat, '', 'target_string')


def remove_redundant_keys(file_name, file_name_write):
    """
    :param file_name:  directory of original data downloaded directly from the annotation website
                  eg.  './data/case-data-revised.json'

    :param file_name_write:  directory of processed data
                  eg.  './data/case-data-revised-clean.json'
    :return: None
    """
    data = json.load(open(file_name, encoding='utf-8'))
    term_data = data['term']

    for term in term_data:
        del term['id']
        del term['createTime']
        del term['status']
        del term['userName']
        del term['userID']
        term['content'] = term['content'][0]['content']
        term['result'] = term['result'][0]['answer']
        for res in term['result']:
            del res['time']

    with open(file_name_write, 'w', encoding='utf-8') as f:
        json.dump(term_data, f, indent=4, ensure_ascii=False)


def remove_extra_returns(content_string):
    content_list = content_string.split('\n')

    while '' in content_list:
        content_list.remove('')

    content_string_clean = '\n\n'.join(content_list)

    return content_string_clean


def get_ah(content_string):
    content_line_list = content_string.split('\n')
    ah = None
    for line in content_line_list[:10]:
        if len(line) < 70 and len(ah_pat.findall(line)) > 0:
            ah = ah_pat.findall(line)[0]
            ah = ah.replace("(", "（").replace(")", "）")
            break
    return ah


def ah2id(ah):
    url = "http://qingfa.fajuhe.com/home/Search/getCaseIdByCaseNo"
    data = {"case_no": ah}
    data = urllib.parse.urlencode(data)
    data = data.encode()
    response = request.urlopen(url=url, data=data)
    response = json.loads(response.read().decode())
    return response['case_id']


def split(case_id):
    url = "http://qingfa.fajuhe.com/home/caseDetail"
    case_id = {'case_id': case_id}
    data = urllib.parse.urlencode(case_id)
    data = data.encode()
    response = request.urlopen(url=url, data=data)
    ret = json.loads(response.read().decode())
    return ret


def get_split_success_and_fail(file_name):
    case_data = json.load(open('./data/case-data-with-case_no-clean.json', encoding='utf-8'))

    split_success = []
    split_fail = []

    for data in tqdm(case_data):
        data_tmp = {}
        if data['caseNO'] is not None:
            case_id = ah2id(data['caseNO'])
            split_tmp = split(case_id)
            if split_tmp['msg'] == 'success':
                data_tmp['termID'] = data['termID']
                data_tmp['caseNO'] = data['caseNO']
                data_tmp['caseID'] = case_id
                data_tmp['content'] = data['content']
                data_tmp['split'] = split_tmp
                data_tmp['result'] = data['result']
                split_success.append(data_tmp)
            else:
                split_fail.append(data)
        else:
            split_fail.append(data)
    return split_success, split_fail


def get_ir_data(file_name='./data/split_success.json'):
    all_data = json.load(open(file_name, encoding='utf-8'))
    ir_data = []
    for data in all_data:
        # add case description label
        result = data['result']

        level1 = []
        level2 = []
        level3 = []

        for r in result:
            label = r['value']
            level1 += label[0:1]
            level2 += label[1:2]
            level3 += label[2:3]

        data['label'] = {'level1': list(set(level1)),
                         'level2': list(set(level2)),
                         'level3': list(set(level3))}

        # other attributes
        attrs = data['split']['info']['attrs']
        data['title'] = attrs['title']
        data['judgement_date'] = attrs['judgement_date']
        data['court_name'] = attrs['court_name']
        data['main_judge'] = attrs['main_judge']

        law_cases = data['split']['info']['law_cases']
        law_cases_new = []
        for lc in law_cases:
            lc_tmp = {}
            lc_tmp['law_name'] = lc['law_name_new']
            lc_tmp['law_clause'] = lc['law_clause']
            lc_tmp['text'] = lc['text']
            law_cases_new.append(lc_tmp)
        data['law_case'] = law_cases_new

        del data['caseID']
        del data['split']

        ir_data_tmp = {}
        ir_data_tmp['termID'] = data['termID']
        ir_data_tmp['title'] = data['title']
        ir_data_tmp['caseNO'] = data['caseNO']
        ir_data_tmp['content'] = data['content']
        ir_data_tmp['label'] = data['label']
        ir_data_tmp['court_name'] = data['court_name']
        ir_data_tmp['main_judge'] = data['main_judge']
        ir_data_tmp['judge_date'] = data['judgement_date']
        ir_data_tmp['law_case'] = data['law_case']
        ir_data_tmp['result'] = data['result']
        ir_data.append(ir_data_tmp)

    return ir_data


def split_train_and_test(filename='./data/chaming/chaming.json'):
    all_data = json.load(open(filename, encoding='utf-8'))
    train_indices = json.load(open('./data/rest/train_indices.json'))
    test_indices = json.load(open('./data/rest/test_indices.json'))

    short = []
    train = []
    test = []

    for data in all_data:
        fact = data['fact']
        for i, f in enumerate(fact):
            if '一审法院认为' in f:
                fact[i] = f[0:f.find('一审法院认为')]
            fact[i] = re.sub(date_pat, '', fact[i])
            fact[i] = re.sub(star_pat, '', fact[i])
            fact[i] = re.sub(prefix_pat, '', fact[i])
            if ('本院' in f or '二审' in f) and '一审' in f and ('予以确认' in f or '一致' in f or '相同' in f) and '另查明' not in f:
                if len(f) <= 50:
                    short.append(f)
                    fact[i] = ''
            while '' in fact:
                fact.remove('')
        if data['termID'] in train_indices:
            train.append(data)
        elif data['termID'] in test_indices:
            test.append(data)

    with open('./data/chaming/chaming_train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, indent=4, ensure_ascii=False)

    with open('./data/chaming/chaming_test.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    all_data = json.load(open('./data/chaming/chaming.json', encoding='utf-8'))
    train_indices = json.load(open('./data/rest/train_indices.json'))
    test_indices = json.load(open('./data/rest/test_indices.json'))

    train, test = [], []

    for data in all_data:
        sentences = []
        sentences_label = []
        fact = data['fact']
        result = data['result']
        for i, f in enumerate(fact):
            if '一审法院认为' in f:
                fact[i] = f[0:f.find('一审法院认为')]
            for sent in fact[i].split('。'):
                if sent != '':
                    sent_label_tmp = []
                    sent_clean = re.sub(punt_pat, '', sent)
                    for r in result:
                        contents = r['content'].split('。')
                        for c in contents:
                            content_clean = re.sub(punt_pat, '', c)
                            if content_clean in sent_clean and content_clean != '':
                                if len(r['value'][2:3]) != 0:
                                    sent_label_tmp.append(r['value'][2:3][0])

                    if len(sent_label_tmp) == 0:
                        sent_label_tmp = ['other']
                    sentences.append({'sentence': sent, 'label': list(set(sent_label_tmp))})
        data['sentences'] = sentences
        sentence_labels = []
        for s in sentences:
            labels = s['label']
            for l in labels:
                if l not in sentence_labels:
                    sentence_labels.append(l)
        data['sent_labels'] = sentence_labels
        if data['termID'] in train_indices:
            train.append(data)
        elif data['termID'] in test_indices:
            test.append(data)


