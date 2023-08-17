import configparser
from icrawler.builtin import BingImageCrawler
import sys
import os
import shutil
import glob
import dlib
import cv2
import random

class Scraping():
    def __init__(self,dir_path,keyword,num_scraping):
        """コンストラクタ

        Args:
            dir_path (str): 収集した画像の出力先フォルダパス
            keyword (str): 検索キーワード
            num_scraping (int): 収集する画像枚数
        """        
        try:
            self.dir_path=dir_path
            self.keyword=keyword
            self.num_scraping=num_scraping
        except Exception as initError:
            print("*** init Error ***")
            print(initError)
            sys.exit()
        self.initialized=True
    def scraping(self):
        """スクレイピングするメンバ関数
        """        
        if self.initialized:
            crawler=BingImageCrawler(downloader_threads=4,storage={'root_dir': self.dir_path})
            crawler.crawl(keyword=self.keyword, max_num=self.num_scraping)
    def cripping(self):
        """顔部分切り抜きするメンバ関数
        """
        detector=dlib.get_frontal_face_detector()
        images=glob.glob(self.dir_path+r"/*")
        for image_path in images:
            img=cv2.imread(image_path)
            faces=detector(img,1) 
            for k,d in enumerate(faces):
                x1=int(d.left())
                y1=int(d.top())
                x2=int(d.right())
                y2=int(d.bottom())
                im=img[y1:y2, x1:x2]
                #224x224にリサイズ
                try:
                    im=cv2.resize(im, (224,224))
                except:
                    continue
                # 切り抜き画像保存
                faceImg_path=image_path.replace("faceImages","faceImages")
                faceImg_dir=os.path.dirname(faceImg_path)
                if not os.path.exists(faceImg_path):
                    os.makedirs(faceImg_dir,exist_ok=True)
                cv2.imwrite(faceImg_path, im)
                # 切り抜き+左右反転画像保存
                flip_faceImg_path=faceImg_path.replace(".jpg","_fliped.jpg")
                cv2.imwrite(flip_faceImg_path,cv2.flip(im,1)) 
        return faceImg_dir

def make_train_data(faceImg_dir):
    train_dir="train_data/train"
    val_dir="train_data/val"
    if not os.path.exists(train_dir):
        os.makedirs(train_dir,exist_ok=True) 
    if not os.path.exists(val_dir):
        os.makedirs(val_dir,exist_ok=True)

    faceImages=glob.glob(faceImg_dir+r"/*")
    random.shuffle(faceImages, random=random.random)
    num_faceImages=len(faceImages)

    for idx,faceImage in enumerate(faceImages):
        img=cv2.imread(faceImage)
        # 学習:検証=8:2
        if idx<num_faceImages*0.8:
            trainImg_path=faceImage.replace("faceImages","train_data/train")
            trainImg_dir=os.path.dirname(trainImg_path)
            if not os.path.exists(trainImg_path):
                os.makedirs(trainImg_dir,exist_ok=True)
            cv2.imwrite(trainImg_path,img)
        else:
            valImg_path=faceImage.replace("faceImages","train_data/val")
            valImg_dir=os.path.dirname(valImg_path)
            if not os.path.exists(valImg_path):
                os.makedirs(valImg_dir,exist_ok=True)
            cv2.imwrite(valImg_path,img)                 
            
def get_config(conf_path):
    """configファイルからパラメータを取得する関数

    Args:
        conf_path (str): configファイルのパス

    Raises:
        Exception: カテゴリ数と検索キーワード数or出力先フォルダパスの数が一致していない例外

    Returns:
        int: カテゴリ数
        int: 収集する画像枚数
        list: 検索キーワード
        list: 出力先フォルダパス
        int: scraping有効フラグ
        int: cripping_flag有効フラグ
    """    
    config=configparser.ConfigParser()
    try:
        config.read(conf_path,encoding='utf-8')
        read_setting=config["setting"]
        categories=int(read_setting["categories"])
        num_scraping=int(read_setting["num_scraping"])
        # eval()がないとlistではなく、stringで読み込まれる 
        keywords=eval(read_setting["keywords"])
        dirpaths=eval(read_setting["dirpaths"])
        scraping_flag=int(read_setting["scraping_flag"])
        cripping_flag=int(read_setting["cripping_flag"])
        if categories!=len(keywords) or categories!=len(dirpaths):
            raise Exception("Not match categories and keywords or dirpaths.")
        return categories,num_scraping,keywords,dirpaths,scraping_flag,cripping_flag
    except Exception as configError:
        print("*** get_config Error ***")
        print(configError)
        sys.exit()

if __name__=='__main__':
    # configファイルからパラメータ取得
    conf_path="./conf/scraping.ini"
    categories,num_scraping,keywords,dirpaths,scraping_flag,cripping_flag=get_config(conf_path=conf_path)

    # 画像収集実行
    if scraping_flag:
        for dirpath,keyword in zip(dirpaths,keywords):
            if os.path.exists(dirpath):
                shutil.rmtree(dirpath)
            ScrapingObject=Scraping(
                dir_path=dirpath,
                keyword=keyword,
                num_scraping=num_scraping
                )
            ScrapingObject.scraping()
            print(f"{keyword} finished.\n")
            del ScrapingObject
    
    # 顔部分切り抜き実行
    if cripping_flag and scraping_flag:
        faceImg_dirs=[]
        try:
            for dirpath,keyword in zip(dirpaths,keywords):
                if not os.path.exists(dirpath):
                    raise Exception(f"Not found: {dirpath}. Please run scraping.")
                else:
                    ScrapingObject=Scraping(
                        dir_path=dirpath,
                        keyword=keyword,
                        num_scraping=num_scraping
                        )
                    faceImg_dir=ScrapingObject.cripping()
                    faceImg_dirs.append(faceImg_dir)
                    del ScrapingObject
        except Exception as crippingError:
            print("*** cripping Error ***")
            print(crippingError)
            sys.exit()

    # 学習用データセット作成実行
    faceImg_dirs=[
        "faceImages/yoda_yuuki",
        "faceImages/tamura_mayu",
        "faceImages/kaki_haruka",
        "faceImages/moriya_rena",
        "faceImages/tamura_hono",
        "faceImages/kobayashi_yui",
        "faceImages/saito_kyoko",
        "faceImages/kato_shiho",
        "faceImages/takamoto_ayaka",
        "faceImages/kosaka_nao"
    ]
    shutil.rmtree("./train_data")
    for faceImg_dir in faceImg_dirs:
        make_train_data(faceImg_dir)
        print(f"faceImages: {faceImg_dir} finished.")