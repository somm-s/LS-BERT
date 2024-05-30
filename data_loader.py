import zipfile
import tqdm

def load_pretrain_data(pre_train_prefixes, config, dataset_path):
    corpus = []
    is_bidirectional = "di-bi" in config
    for prefix in pre_train_prefixes:
        year_corpus = []
        prefix_dataset_path = dataset_path + "/" + prefix + "_" + config + ".zip"
        print("Loading data from " + prefix_dataset_path)
        
        with zipfile.ZipFile(prefix_dataset_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            for file_name in tqdm.tqdm(file_list):
                if (not file_name.endswith('.csv')):
                    continue
                name = file_name.split('/')[-1]
                name = name.split('.csv')[0]
                with zip_ref.open(file_name) as file:
                    content = file.read().decode('utf-8')

                if is_bidirectional:
                    document = []
                    for line in content.split('\n'):
                        if not line:
                            continue
                        line = line.strip()
                        document.append(line)
                    year_corpus.append(document)
                else:
                    documents = {}                    
                    for line in content.split('\n'):
                        if not line:
                            continue
                        line = line.strip()
                        ip, sentence = line.split(',')
                        if ip in documents:
                            documents[ip].append(sentence)
                        else:
                            documents[ip] = [sentence]
                    for ip in documents:
                        year_corpus.append(documents[ip])
        corpus += year_corpus
    return corpus


def load_train_data(prefix, config, dataset_path):

    is_bidirectional = "di-bi" in config

    with open(dataset_path + "/" + prefix + "_ip_labels.txt", 'r') as file:
        ip_labels = file.readlines()
    ip_labels = [x.strip() for x in ip_labels]
    benign_ips = set()
    rt_ips = set()
    for ip_label in ip_labels:
        ip, label = ip_label.split(',')
        if label == '1':
            benign_ips.add(ip)
        if label == '0':
            rt_ips.add(ip)

    benign_corpus = []
    rt_corpus = []
    prefix_dataset_path = dataset_path + "/" + prefix + "_" + config + ".zip"
    print("Loading data from " + prefix_dataset_path)
    
    with zipfile.ZipFile(prefix_dataset_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        for file_name in tqdm.tqdm(file_list):
            if (not file_name.endswith('.csv')):
                continue
            name = file_name.split('/')[-1]
            name = name.split('.csv')[0]
            
            if is_bidirectional:
                ip1 = name.split('-')[0]
                ip2 = name.split('-')[1]
                if ip1 not in benign_ips and ip1 not in rt_ips and ip2 not in benign_ips and ip2 not in rt_ips:
                    continue
                document = []
                with zip_ref.open(file_name) as file:
                    content = file.read().decode('utf-8')
                for line in content.split('\n'):
                    if not line:
                        continue
                    line = line.strip()
                    document.append(line)
                
                if ip1 in benign_ips or ip2 in benign_ips and ip1 not in rt_ips and ip2 not in rt_ips:
                    benign_corpus.append(document)
                if ip1 in rt_ips or ip2 in rt_ips and ip1 not in benign_ips and ip2 not in benign_ips:
                    rt_corpus.append(document)
                
            else:
                if name not in benign_ips and name not in rt_ips:
                    continue
                with zip_ref.open(file_name) as file:
                    content = file.read().decode('utf-8')
                    
                documents = {}
                for line in content.split('\n'):
                    if not line:
                        continue
                    line = line.strip()
                    ip, sentence = line.split(',')
                    if ip in documents:
                        documents[ip].append(sentence)
                    else:
                        documents[ip] = [sentence]
                        
                document = []
                for ip in documents:
                    document.append(documents[ip])
                    
                if name in benign_ips:
                    benign_corpus.append(document)
                if name in rt_ips:
                    rt_corpus.append(document)
    return benign_corpus, rt_corpus