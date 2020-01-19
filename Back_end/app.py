from flask import Flask, request, Response,jsonify
from mysqlClass import mySQL
import pandas as pd
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


@app.route('/num_counting_type/', methods=['POST', 'GET'])
def numInAttributes():
    requestArgs = request.values
    db = mySQL(host, user, pwd, dbname)
    sql = """select DISTINCT course_en.course_type
        from course_en"""
    # print(db.searchByAttributes(sql))
    tuple1 = db.searchByAttributes(sql)
    # for i in tuple1.__str__():
    #     if (tuple1.__str__())
    str1 = tuple1.__str__()
    print(jsonify(str1))
    return jsonify(str1)

@app.route('/searchbytype/')
def searchByType():
    requestArgs = request.values
    db = mySQL(host, user, pwd, dbname)
    sql = """select *
        from course_en where course_en.course_type='{0}'""".format(requestArgs)
    # print(db.searchByAttributes(sql))
    tuple1 = db.searchByAttributes(sql)
    str1 = tuple1.__str__()
    return str1

def createTable():
    sql = """CREATE IF NOT EXISTS 'relation_theme_id' ('Theme' varchar(50) not null, 'title' minitext()) private key 'Theme'"""
    data1 = pd.read_csv('output.csv', error_bad_lines=False, encoding='ISO-8859-1')
    data2 = pd.read_csv('title.csv', error_bad_lines=False, encoding='ISO-8859-1')
    list1 = []
    for i in data2:
        for j in data1:
            if (j > 0.6):
                list.append(i, )

# def create():
    # sql = """CREATE TABLE IF NOT EXISTS 'test2'(
    # 'Name' varchar(20) not null, 'Id' int not null) PRIMARY KEY ('ID')"""
    # db.createTable(sql)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
