
import base64
import logging
import json
import boto3
#import numpy
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

print('Loading Lambda function')

runtime=boto3.Session().client('sagemaker-runtime')
endpoint_Name='dog-breed-classifier'

def lambda_handler(event, context):
    """
    Test the lambda function with the following event:
    {
    "body": {
        "url": "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/20113314/Carolina-Dog-standing-outdoors.jpg"
    }
    }
    """

    print('Context:::',context)
    print('EventType::',type(event))
    print("Event", event)
    print("Body", event.get("body"))
    print("Body Type", type(event.get("body")))
    bs=event.get("body")

    if isinstance(bs, str):
        print("-> Converting string body to json.")
        bs = json.loads(bs)

    runtime=boto3.Session().client('sagemaker-runtime')
    
    response=runtime.invoke_endpoint(EndpointName=endpoint_Name,
                                    ContentType="application/json",
                                    Accept='application/json',
                                    #Body=bytearray(x)
                                    Body=json.dumps(bs))
    
    result=response['Body'].read().decode('utf-8')
    sss=json.loads(result)
    
    return {
        'statusCode': 200,
        'headers' : { 'Content-Type' : 'application/json', 'Access-Control-Allow-Origin' : '*' },
        'body' : json.dumps(sss)
        }