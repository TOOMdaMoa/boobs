import csv
import psycopg2
import time
from config import config


def get_csv_data(input_filename) -> [[str]]:
    data: [[str]] = []
    with open(input_filename, newline='', encoding='latin-1') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if len(row) > 0:
                data.append(row)

    return data

def create_table():
    data = get_csv_data("./data import/data.csv")
    header = data[0]
    data = data[1:]

    # replace '' with NULL
    for row in range(0, len(data)):
        for value in range(0, len(data[row])):
            if data[row][value] == '':
                data[row][value] = None

    sql = "CREATE TABLE user_views("+header[0] + " text"
    for i in range(1, len(header)):
        sql += ", " + header[i] + " text"
    sql += ")"

    print(sql)

    conn = None
    try:
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        cur.execute(sql)
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

def fill_table(table_name):
    data = get_csv_data("./data import/data.csv")
    header = data[0]
    data = data[1:]

    # replace '' with NULL
    for row in range(0, len(data)):
        for value in range(0, len(data[row])):
            if data[row][value] == '':
                data[row][value] = None

    sql = "INSERT INTO {}(".format(table_name)
    headers_join = ", "
    headers_join = headers_join.join(header)
    sql += headers_join + ") VALUES(%s"
    for i in range(1, len(header)):
        sql += ",%s"
    sql += ")"
    print(sql)

    conn = None
    try:
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        cur.executemany(sql, data)
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

if __name__ == '__main__':
    inspection_processing_start = time.time()
    fill_table("user_views")
    processing_duration_seconds = int(time.time() - inspection_processing_start)
    print("Trvalo to {} sekund, tedy {} hodin {} minut {} sekund.".format(processing_duration_seconds,
                                                                          int(processing_duration_seconds / 3600),
                                                                          int((
                                                                                      processing_duration_seconds % 3600) / 60),
                                                                          int((
                                                                                      processing_duration_seconds % 3600) % 60)))
    print('Hotovo')