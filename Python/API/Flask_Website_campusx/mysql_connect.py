import pandas as pd
import mysql.connector
import getpass,os


class MySqlConnection:
    def __init__ (self,host,user,database):
        self.host = host
        self.user = user
        self.password =  self.password = getpass.getpass(prompt="Enter your password: ")
        self.database = database
        self.conn = None
        

    
    def connect(self):
        """Establish Mysql Connection.."""
        try:
            self.conn = mysql.connector.connect(host = self.host,
                                                user = self.user,
                                                password = self.password,
                                                database = self.database,)
            if self.conn.is_connected():
                print(f"MySql Connection Details:\nData Base --> {self.database} ")

                cursor = self.conn.cursor()
                cursor.execute("select user();")
                print(f"User --> {cursor.fetchone()[0]}")

                cursor.execute("SELECT DATE_FORMAT(NOW(), '%Y-%m-%d %r') AS CurrentDateTime;")
                print(f"Date Time --> {cursor.fetchone()[0]}")
    
        except Exception as e:
            print(f"MySql Connect failed .....{e}")
            self.conn = None

            
    def table_query(self,query):
        """Get records from Mysql into the Dataframe"""
        try:
            if self.conn is None or not self.conn.is_connected():
                print("Reconnecting to MySQL...")
                self.connect()
                
            cursor = self.conn.cursor()
            cursor.execute(query)
            records = cursor.fetchall()
            cursor.close()
            
            column_list =[desc[0] for desc in  cursor.description]
            df = pd.DataFrame(records, columns=column_list)
            return df
        except  mysql.connector.Error as e:
            print(f"Error reading data from MySQL: {e}")
            return None
            
    
    def create_table(self, create_table_query, table_name):
        """Create a table in the database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(create_table_query)
            print(f"Table [{table_name}] created successfully.")
            cursor.close()
        except Exception as e:
            print(f"Error creating table: {e}")


    def insert_record_from_df (self, table_name, df):
        try:  
            if self.conn is None or not self.conn.is_connected():
                print("Reconnecting to MySQL...")
                self.connect()
            cursor = self.conn.cursor()
            for i, row in df.iterrows():
                # Create an INSERT query for each row
                row_values = row.astype(str).values.tolist()
                
                placeholders = ", ".join(["%s"] * len(row))
                columns = ", ".join(row.index)
                insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                cursor.execute(insert_query, tuple(row_values))
            self.conn.commit()
            print(f"Inserted {len(df)} rows into {table_name} successfully.")
            cursor.close()
        except Exception as e:
            print(f"Error inserting data: {e}")
            self.conn.rollback()
            cursor.close()

    
    def connection_close(self):
        if self.conn:
            self.conn.close()
            print('MySql Connection Closed.........\n')
 
        

    
 
        

