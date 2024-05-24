import zipfile
import tqdm

def load_pretrain_data(years, config, dataset_path):
    corpus = []
    is_bidirectional = "di-bi" in config
    for year in years:
        year_corpus = []
        year_dataset_path = dataset_path + "/yr-" + str(year) + "_" + config + ".zip"
        print("Loading data from " + year_dataset_path)
        
        with zipfile.ZipFile(year_dataset_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            for file_name in tqdm.tqdm(file_list):
                if (not file_name.endswith('.csv')):
                    continue
                name = file_name.split('/')[-1]
                name = name.split('.csv')[0]
                with zip_ref.open(file_name) as file:
                    content = file.read().decode('utf-8')  # Assuming the content is in UTF-8 encoding

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


def load_short_pairs(year, config, dataset_path):
    year_dataset_path = dataset_path + "/yr-" + str(year) + "_" + config + ".zip"
    
    sentence_corpus = []
    metadata_corpus = []
    
    with zipfile.ZipFile(year_dataset_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        for file_name in tqdm.tqdm(file_list):
            if (not file_name.endswith('.csv')):
                continue
            name = file_name.split('/')[-1]
            ip1 = name.split('.csv')[0]
            with zip_ref.open(file_name) as file:
                content = file.read().decode('utf-8')

            last_sentences = {}
            documents = {}
            for line in content.split('\n'):
                if not line:
                    continue
                line = line.strip()
                ip2, sentence = line.split(',')
                if ip2 not in last_sentences:
                    last_sentences[ip2] = sentence
                    continue
                
                sentence_pair = (last_sentences[ip2], sentence)
                if ip2 in documents:
                    documents[ip2].append(sentence_pair)
                else:
                    documents[ip2] = [sentence_pair]
                    
            if documents:
                metadata = {"ip_name": ip1}
                metadata["documents"] = {}
                for ip2 in documents:
                    metadata["documents"][ip2] = {
                        "sentence_begin_idx": len(sentence_corpus),
                        "sentence_end_idx": len(sentence_corpus) + len(documents[ip2]),
                    }
                    sentence_corpus += documents[ip2]
                metadata_corpus.append(metadata)
    return sentence_corpus, metadata_corpus


def load_documents(year, config, dataset_path):
    year_dataset_path = dataset_path + "/yr-" + str(year) + "_" + config + ".zip"
    
    sentence_corpus = []
    metadata_corpus = []
    
    with zipfile.ZipFile(year_dataset_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        for file_name in tqdm.tqdm(file_list):
            if (not file_name.endswith('.csv')):
                continue
            name = file_name.split('/')[-1]
            ip1 = name.split('.csv')[0]
            with zip_ref.open(file_name) as file:
                content = file.read().decode('utf-8')

            metadata = {"ip_name": ip1}
            metadata["documents"] = {}
            documents = {}
            for line in content.split('\n'):
                if not line:
                    continue
                line = line.strip()
                ip2, sentence = line.split(',')
                if ip2 in documents:
                    documents[ip2].append(sentence)
                else:
                    documents[ip2] = [sentence]
                    
            for ip2 in documents:
                metadata["documents"][ip2] = {
                    "sentence_begin_idx": len(sentence_corpus),
                    "sentence_end_idx": len(sentence_corpus) + len(documents[ip2]),
                }
                sentence_corpus += documents[ip2]
            metadata_corpus.append(metadata)
    return sentence_corpus, metadata_corpus


def load_train_data(year, config, dataset_path):

    is_bidirectional = "di-bi" in config

    with open("ip_labels_" + str(year) + ".txt", 'r') as file:
        ip_labels = file.readlines()
    ip_labels = [x.strip() for x in ip_labels]
    benign_ips = set()
    rt_ips = set()
    for ip_label in ip_labels:
        ip, label = ip_label.split(',')
        if label == '3':
            benign_ips.add(ip)
        if label == '2':
            rt_ips.add(ip)

    benign_corpus = []
    rt_corpus = []
    year_dataset_path = dataset_path + "/yr-" + str(year) + "_" + config + ".zip"
    print("Loading data from " + year_dataset_path)
    
    with zipfile.ZipFile(year_dataset_path, 'r') as zip_ref:
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


def load_train_data_2024(year, config, dataset_path):

    is_bidirectional = "di-bi" in config

    with open("ip_labels_" + str(year) + ".txt", 'r') as file:
        ip_labels = file.readlines()
    from ip_utils import expand_ipv6
    ip_labels = [x.strip() for x in ip_labels]
    ip_labels = [expand_ipv6(ip) for ip in ip_labels]
    benign_ips = set()
    rt_ips = set()
    for ip_label in ip_labels:
        rt_ips.add(ip_label)

    benign_corpus = []
    rt_corpus = []
    year_dataset_path = dataset_path + "/yr-" + str(year) + "_" + config + ".zip"
    print("Loading data from " + year_dataset_path)
    
    with zipfile.ZipFile(year_dataset_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        for file_name in tqdm.tqdm(file_list):
            if (not file_name.endswith('.csv')):
                continue
            name = file_name.split('/')[-1]
            name = name.split('.csv')[0]
            
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
                
            if name in rt_ips:
                rt_corpus.append(document)
            else:
                benign_corpus.append(document)
    return benign_corpus, rt_corpus