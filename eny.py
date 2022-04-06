host = '157.230.209.171'
username = 'innis_1666'
user = username
password = 'wy1hPZc9W4uN3eLRfdlU7YLT5wrBz8eJ'

def get_db_url(database, user=user, password=password, host=host):
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    return url
