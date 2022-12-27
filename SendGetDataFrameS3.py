import io
import boto3
import configS3
import pandas as pd
from SendGetDataFrameS3 import *

def sendDataframeToS3(df, file_name, location = "local"):

    if location == "s3":
        # create s3 client
        s3_resource = boto3.client('s3',
                    aws_access_key_id = configS3.key,
                    aws_secret_access_key = configS3.secret_key)
        
        # save DataFrame to S3
        with io.StringIO() as csv_buffer:
            df.to_csv(csv_buffer, index=False)
            response = s3_resource.put_object(
                Bucket=configS3.bucket1, Key=file_name, Body=csv_buffer.getvalue())
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        
        # response
        if status == 200:
            print(f"Successful S3 put_object response. Status - {status}")
            success = True
        else:
            print(f"Unsuccessful S3 put_object response. Status - {status}")
            success = False

    elif location == "local":
        df.to_csv(file_name, index=False)
        success = True
    
    return success

def fetchDataframeFromS3(file_name, location = "local"):

    if location == "local":
        try:
            df = pd.read_csv(file_name)
        except:
            print("Fail to load the file ")
            pass
    elif location == "s3":    
        # create s3 client
        s3_resource = boto3.client('s3',
                    aws_access_key_id = configS3.key,
                    aws_secret_access_key = configS3.secret_key)
        
        # save DataFrame to S3
        response = s3_resource.get_object(Bucket=configS3.bucket1, Key=file_name)
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        
        # response
        if status == 200:
            print(f"Successful S3 get_object response. Status - {status}")
            df = pd.read_csv(response.get("Body"))
            print(df)
            #df = fetchDataframeFromS3(response.get("Body"))
            success = True
        else:
            print(f"UNsuccessful S3 get_object response. Status - {status}")
            df = None
            success = False
    
    return df #, success

"""df = pd.DataFrame(data = {"A": [0, 1], "B": [2,3]})
print("initial df:", df)

success = sendDataframeToS3(df, "df3.csv")
df = fetchDataframeFromS3("df3.csv")
print("retreive df:", df)"""
