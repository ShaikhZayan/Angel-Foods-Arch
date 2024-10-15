from fastapi import FastAPI, Depends, HTTPException
from typing import Annotated
from jose import JWTError, jwt 
from datetime import timedelta, datetime

ALGORITHM = "HS256"
SECRET_KEY = "Angle Food Secret Key"

def create_access_token(subject: str) -> str:
    to_encode = {"sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_decode_token(access_token:str):
    decoded_jwt = jwt.decode(access_token, SECRET_KEY, algorithms=ALGORITHM)
    return decoded_jwt
