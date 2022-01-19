import sys
import os
import dvc.api
import pickle
import train
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def retrain():
    return train.main(log = True)

def main():
    print("New data is available")
    print("Doing retraining.......")
    model = retrain()
    print("Done!")

    upload_model(model)

def upload_model(model):
    pickle.dump(model, open("model.pkl", 'wb'))

    gauth = GoogleAuth()       
    gauth.LoadCredentialsFile('../src/credentials.json')
    drive = GoogleDrive(gauth)  
    
    upload_file = 'model.pkl'
    gfile = drive.CreateFile({'parents': [{'id': '1BnBxVTuQ8Otab2h7aQFdXI14Ou3uIVVk'}]})
    
    gfile.SetContentFile(upload_file)
    gfile.Upload()

if __name__ == "__main__":
    main()
    
    
    
    
    