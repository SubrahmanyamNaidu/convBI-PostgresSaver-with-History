import psycopg
from psycopg import OperationalError

def test_db_connection(request):
    try:
        connection = psycopg.connect("postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable")
        cursor = connection.cursor()
        cursor.execute("SELECT version();")
        print("Connection successful!")
        connection.close()
        return True
    except OperationalError as e:
        print(f"Connection failed: {e}")
        return False
    

if __name__=="__main__":
    request={
            "host":"localhost",
            "port":"5432",
            "database_name":"postgres",
            "username":"postgres",
            "password":"postgres"
    }
    test_db_connection(request)