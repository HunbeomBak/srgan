from google_drive_downloader import GoogleDriveDownloader as gdd
import os



def train_dataset():
    save_path = './data/CelebA-HQ_Dataset/Train/Train.zip'
    gdd.download_file_from_google_drive(file_id='1CF6Jkk9sK6eevBT3h3SsZLYqKTQMEDlR',
                                    dest_path=save_path,
                                    showsize=True,
                                    unzip=True)
    os.remove(save_path)  
    
def val_dataset():
    save_path = './data/CelebA-HQ_Dataset/Val/Val.zip'
    gdd.download_file_from_google_drive(file_id='1gg1qQm_Ms90qxgM1gSVOIbnHi7v7uHaH',
                                    dest_path=save_path,
                                    showsize=True,
                                    unzip=True)
    os.remove(save_path) 
    
    
def down_dataset():
    train_dataset()
    val_dataset()


def down_model():
    pass

if __name__ == "__main__":
    down_dataset()