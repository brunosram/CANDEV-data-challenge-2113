import MySQLdb
class mySQL():
    # db = MySQLdb.connect("localhost", "root", "root", "cdev", charset="utf8")
    def __init__(self, host, user, password, dBName):
        self.host = host
        self.user = user
        self.password = password
        self.dBName = dBName
        self.db = self.connectDB()
        if (self.db):
            self.cursor = self.db.cursor()

    def connectDB(self):
        try:
            db = MySQLdb.connect(self.host, self.user, self.password, self.dBName, charset="utf8")
        except:
            print("Fail to connect!")
        else:
            return db

    def closeDB(self):
        if (self.db):
            try:
                self.cursor.close()
                self.db.close()
            except:
                print("Fail to close")

    # return the number for different theme
    def searchByAttributes(self, sql):
        # db = MySQLdb.connect("localhost",  "root", "root", "cdev", charset="utf8")
        try:
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            return result
        except:
            self.db.rollback()
            return "Fail for searching"

    def createTable(self, sql):
        try:
            self.cursor.execute(sql)
        else:
            self.db.rollback()

    # def searchByType(self):
        # sql = """SELECT course_en.course_code from course_en where course_en.course_type"""

    # def searchByID(self):
    #     sql = """"""

    # including insert
    def updateCourse(self, sql):
        # sql = """"""
        try:
            self.cursor.execute(sql)
            self.db.commit()
            return "success"
        except:
            self.db.rollback()
            return "Fail for updating"

    # def creatTable()
    # def deleteCourse(self):
    #     sql = """"""



    # def test():
    #     db = MySQLdb.connect("localhost", "root", "root", "cdev", charset="utf8")
    #     sql = """SELECT
    #         COUNT(test.coursecode)
    #     FROM
    #         test
    #     WHERE
    #         test.type = 'Online'
    #     """
    #
    #     cursor = db.cursor(cursorclass=MySQLdb.cursors.DictCursor)
    #     try:
    #         cursor.execute(sql)
    #         result = cursor.fetchone()
    #         # print(result)
    #         return result
    #     except:
    #         db.rollback()
    #
    #     db.close()