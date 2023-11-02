from torchvision import transforms
from train_utils import create_dataset
try:
    import ruamel.yaml as yaml
except BaseException:
    import ruamel_yaml as yaml

def len_(x):
    if x is None:
        return 0
    return len(x)

def check(data_config):
    print(data_config)
    data_config = yaml.load(open(data_config, 'r'), Loader=yaml.Loader)
    train_transform = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    train_dataset, dev_dataset, test_dataset = create_dataset(data_config, train_transform, train_transform)
    print(len_(train_dataset), len_(dev_dataset), len_(test_dataset))
    # print(train_dataset.)

if __name__ == "__main__":
    check("configs/vqa2021.yml")
    check("configs/vqarad.yml")
    check("configs/slake.yml")
