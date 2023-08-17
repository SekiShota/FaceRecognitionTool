import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision import datasets, models, transforms
import os,sys
import time 
import copy
import configparser
from tqdm import tqdm

from trainlogger import train_logger

class TrainModel():
    """モデル学習クラス
    """    
    def __init__(self, train_dataset_dir, categories, batch_size, epochs, logger):
        """コンストラクタ

        Args:
            train_dataset_dir (str): 学習用画像のフォルダパス
            categories (int): カテゴリ数
            batch_size (int): バッチサイズ
            epochs (int): エポック数
            logger (logging): ロガー
        """        
        self.train_dataset_dir=train_dataset_dir
        self.categories=categories
        self.batch_size=batch_size
        self.epochs=epochs
        self.logger=logger

    def make_dataset(self):
        """データセット作成するメンバ関数

        Returns:
            dataloader: データローダー
            int: データセットに含まれるデータ数
        """        
        # データセット作成
        data_transforms = {
            'train': transforms.Compose([
                transforms.GaussianBlur(kernel_size=3),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        image_datasets = {x: datasets.ImageFolder(os.path.join(train_dataset_dir, x),data_transforms[x]) for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.batch_size,shuffle=True, num_workers=4) for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        return dataloaders, dataset_sizes

    def train_model(self):
        """モデルの学習を実行するメンバ関数

        Returns:
            torch: 学習済みモデルオブジェクト
        """        
        # """学習実行"""
        self.logger.info("train start.")
        # デバイス定義
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # モデル定義
        model_ft=timm.create_model("efficientnet_b0", pretrained=True, num_classes=self.categories)
        model_ft.to(device)
        # 学習用データセット定義
        dataloaders, dataset_sizes=self.make_dataset()
        # 損失関数：クロスエントロピー誤差
        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized、最適化アルゴリズム：確率的勾配降下法
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs、学習率調整
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        since = time.time()
        # ベストモデルのパラメータと正解率
        best_model_wts = copy.deepcopy(model_ft.state_dict())
        best_acc = 0.0

        for epoch in tqdm(range(epochs)):
            self.logger.info('Epoch {}/{}'.format(epoch+1, epochs))
            self.logger.info('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    optimizer_ft.step()
                    exp_lr_scheduler.step()
                    model_ft.train()  # Set model_ft to training mode
                else:
                    model_ft.eval()   # Set model_ft to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer_ft.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model_ft(inputs)
                        #tensor(max, max_indices)なのでpredは0,1のラベル
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                self.logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model_ft
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model_ft.state_dict())

        time_elapsed = time.time() - since
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        self.logger.info('Best val Acc: {:4f}'.format(best_acc))

        # load best model_ft weights
        model_ft.load_state_dict(best_model_wts)
        self.logger.info("train finished.")
        return model_ft

def get_config(conf_path, logger):
    """configファイルからパラメータを取得する関数

    Args:
        conf_path (str): configファイルのパス
        logger (logging): ロガー

    Raises:
        Exception: カテゴリ数とtrain下のフォルダ数 or val下のフォルダ数が一致していない例外

    Returns:
        str: 学習用画像フォルダパス
        int: カテゴリ数
        int: バッチサイズ
        int: エポック数
    """    
    config=configparser.ConfigParser()
    try:
        config.read(conf_path,encoding='utf-8')
        read_setting=config["setting"]
        train_dataset_dir=read_setting["train_dataset_dir"]
        categories=int(read_setting["categories"])
        batch_size=int(read_setting["batch_size"])
        epochs=int(read_setting["epochs"])

        num_train=len(os.listdir(os.path.join(train_dataset_dir,"train")))
        num_val=len(os.listdir(os.path.join(train_dataset_dir,"val")))
        if categories!=num_train or categories!=num_val:
            raise Exception("Not match categories and num_train or num_val.")
        return train_dataset_dir,categories,batch_size,epochs
    
    except Exception as configError:
        logger.warning("*** get_config Error ***")
        logger.warning(configError)
        sys.exit()


if __name__=="__main__":
    # ロガー定義
    logger,d=train_logger()
    # configファイルからパラメータ取得
    config_path="./conf/train.ini"
    train_dataset_dir,categories,batch_size,epochs=get_config(conf_path=config_path, logger=logger)
    # trainオブジェクト定義
    Trainner=TrainModel(
        train_dataset_dir=train_dataset_dir,
        categories=categories,
        batch_size=batch_size,
        epochs=epochs,
        logger=logger
    )

    # training
    try:
        model = Trainner.train_model()
        # モデル保存 model/YYYYMMDDhhmmss/model.pth
        model_dir=os.path.join("./model",d)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path=os.path.join(model_dir,"model.pth")
        print(model_path)
        torch.save(model, model_path)
    except Exception as trainError:
        logger.warning("*** error ***")
        logger.warning(trainError)
        sys.exit()
