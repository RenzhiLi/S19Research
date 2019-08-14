import json


def testLoad():
    with open("train-v2.0.json",'r') as f:
        data = f.read()
    data = json.loads(data)
    docs = []
    querys = []
    n = 0
    for i in range(len(data['data'])):
        doc = ""
        for j in range(len(data['data'][i]['paragraphs'])//3):
            qas = data['data'][i]['paragraphs'][j]['qas']
            for k in range(len(qas)//15):
                if qas[k]['is_impossible'] == False:
                    query = {'q': qas[k]['question'],'docNum': n}
                    # 'docNum':i
                    querys.append(query)
                    doc = {'plainText': qas[k]['question'],'ans':qas[k]['answers'][0]}
                    # doc = doc + data['data'][i]['paragraphs'][j]['context']
                    docs.append(doc)
                    n += 1
    return querys, docs


#print(data['data'][0]['paragraphs'][2]['context'][985:])

# t = "" 
# for i in data['data']:
#     t += i['title'] + " "
# print(t)

if __name__ == '__main__':
    testLoad()


