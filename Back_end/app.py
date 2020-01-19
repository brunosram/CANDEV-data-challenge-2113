from flask import Flask
from mysqlClass import mySQL

app = Flask(__name__)
# app.config.from_object(config)
host = "localhost"
user = "root"
pwd = "root"
dbname = "cdev"
# @app.route('/', methods=['POST'])
# @app.route('/')
# def hello_world():
#     return 'Hello World!'


@app.route('/')
def numInAttributes():
    db = mySQL(host, user, pwd, dbname)
    sql = """select DISTINCT course_en.course_type
        from course_en"""
    # print(db.searchByAttributes(sql))
    tuple1 = db.searchByAttributes(sql)
    str1 = tuple1.__str__()
    # # print(str1)
    # list1 = list(tple)
    # print(list1[0]+list1[1])
    # print(str1[0])
    return str1

def searchByType():
    db = mySQL(host, user, pwd, dbname)
    sql = """select *
        from course_en where course_en.course_type='{0}'""".format("Online")
    # print(db.searchByAttributes(sql))
    tuple1 = db.searchByAttributes(sql)
    str1 = tuple1.__str__()
    # # print(str1)
    # list1 = list(tple)
    # print(list1[0]+list1[1])
    # print(str1[0])
    return str1

# def create():
    # sql = """CREATE TABLE IF NOT EXISTS 'test2'(
    # 'Name' varchar(20) not null, 'Id' int not null) PRIMARY KEY ('ID')"""
    # db.createTable(sql)

if __name__ == '__main__':
    app.run()
