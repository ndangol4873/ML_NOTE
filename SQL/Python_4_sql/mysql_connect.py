
import pandas as pd
import mysql.connector
import getpass,os



class MySqlConnection:
    def __init__ (self,host,user,database=None):
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
                                                password = self.password)
            if self.conn.is_connected():
                print(f"MySql Server Connection Established...\n\nConnection Details:")

                cursor = self.conn.cursor()
                cursor.execute("select user();")
                print(f"User --> {cursor.fetchone()[0]}")

                cursor.execute("SELECT DATE_FORMAT(NOW(), '%Y-%m-%d %r') AS CurrentDateTime;")
                print(f"Date Time --> {cursor.fetchone()[0]}")
    
        except Exception as e:
            print(f"MySql Connect failed .....{e}")
            self.conn = None




    def select_database(self, db_name): 
        """Select a specific database.""" 
        if self.conn is None or not self.conn.is_connected(): 
            print("Reconnecting to MySQL...") 
            self.connect() 
        try: 
            self.conn.database = db_name 
            print(f"Connected to database: {db_name}") 
        except Exception as e:
            print(f"Failed to select database {db_name}: {e}")




    def create_datebase(self,db_name):
        """Create New Database"""
        if self.conn is None:
            print("Please Connect to the MySql Server")
            return 
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"DROP DATABASE IF EXISTS {db_name}") 
            cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"Database {db_name} created successfully")
        except Exception as e:
            print(f"Failed to create database {db_name}: {e}")


    def connect_to_database(self, db_name): 
        """Connect to a specific database.""" 
        self.select_database(db_name)

        

    def sql_dtype_map(self):
       """Constant SQL Data Type Mapping Details"""
       dtype_map_dict = {
                        'float64':'DECIMAL(20, 2)', 
                        'int64':'INTEGER(225)', 
                        'object' : "VARCHAR(225)"  
                        }
       
       return dtype_map_dict
    


    
    def create_tbl_script (self, df, table_name, primary_key=None):
        "Generate Create table Sql Script for any dataframes"
        columns = []
        sql = f"""drop table if exists {table_name}; \nCREATE TABLE {table_name} (\n"""
        for col in df.columns:

            sql_dtype = self.sql_dtype_map()[str(df[col].dtype)]
            if col == primary_key:
                column_syntax = f"{col} {sql_dtype} PRIMARY KEY AUTO_INCREMENT\n"
            else:
                column_syntax = f"{col} {sql_dtype}\n"
            columns.append(column_syntax)
        columns_syntax = ",".join(columns) # Adding new line and indentation for better readability 
        sql +=  columns_syntax + ");" 
        print(sql)
        return sql


 
    
    def create_table(self, create_table_query, table_name):
        """Create a table in the database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(create_table_query)
            print(f"Table [{table_name}] created successfully.")
            # cursor.close()
        except Exception as e:
            print(f"Error creating table: {e}")

    def insert_record (self, insert_script, table_name):
        """Inserts the records from Dataframe to the Table """
        try:  
            if self.conn is None or not self.conn.is_connected():
                print("Reconnecting to MySQL...")
                self.connect()
            cursor = self.conn.cursor()
            cursor.execute(insert_script)
            self.conn.commit()
            print("rows inserted successfully.")
        except Exception as e:
            print(f"Error inserting data: {e}")
            self.conn.rollback()


    def insert_record_from_df (self, table_name, df):
        """Inserts the records from Dataframe to the Table """
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
            # cursor.close()
        except Exception as e:
            print(f"Error inserting data: {e}")
            self.conn.rollback()
            # cursor.close()




    def select_table_query(self,query):
        """Get records from Mysql into the Dataframe"""
        try:
            if self.conn is None or not self.conn.is_connected():
                print("Reconnecting to MySQL...")
                self.connect()
                
            cursor = self.conn.cursor()
            cursor.execute(query)
            records = cursor.fetchall()
            # cursor.close()
            
            column_list =[desc[0] for desc in  cursor.description]
            df = pd.DataFrame(records, columns=column_list)
            return df
        except  mysql.connector.Error as e:
            print(f"Error reading data from MySQL: {e}")
            return None
        


    def connection_close(self):
        if self.conn:
            cursor = self.conn.cursor()
            cursor.execute("SELECT DATE_FORMAT(NOW(), '%Y-%m-%d %r') AS CurrentDateTime;")
            close_datetime = cursor.fetchone()[0]
            self.conn.close()
            print(f"MySql Connection Closed.........{close_datetime}\n")
 
 
        

 
        

