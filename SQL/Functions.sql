

-- Create table If Not Exists Examinations (student_id int, subject_name varchar(20));
-- Create table If Not Exists Students (student_id int, student_name varchar(20));
-- insert into Students (student_id, student_name) values ('1', 'Alice');
-- insert into Students (student_id, student_name) values ('2', 'Bob');
-- insert into Students (student_id, student_name) values ('13', 'John');
-- insert into Students (student_id, student_name) values ('6', 'Alex');

-- Create table If Not Exists Subjects (subject_name varchar(20));
-- insert into Subjects (subject_name) values ('Math');
-- insert into Subjects (subject_name) values ('Physics');
-- insert into Subjects (subject_name) values ('Programming');

-- Create table If Not Exists Examinations (student_id int, subject_name varchar(20));
-- insert into Examinations (student_id, subject_name) values ('1', 'Math');
-- insert into Examinations (student_id, subject_name) values ('1', 'Physics');
-- insert into Examinations (student_id, subject_name) values ('1', 'Programming');
-- insert into Examinations (student_id, subject_name) values ('2', 'Programming');
-- insert into Examinations (student_id, subject_name) values ('1', 'Physics');
-- insert into Examinations (student_id, subject_name) values ('1', 'Math');
-- insert into Examinations (student_id, subject_name) values ('13', 'Math');
-- insert into Examinations (student_id, subject_name) values ('13', 'Programming');
-- insert into Examinations (student_id, subject_name) values ('13', 'Physics');
-- insert into Examinations (student_id, subject_name) values ('2', 'Math');
-- insert into Examinations (student_id, subject_name) values ('1', 'Math');



-- Create table If Not Exists Signups (user_id int, time_stamp datetime);
-- insert into Signups (user_id, time_stamp) values ('3', '2020-03-21 10:16:13');
-- insert into Signups (user_id, time_stamp) values ('7', '2020-01-04 13:57:59');
-- insert into Signups (user_id, time_stamp) values ('2', '2020-07-29 23:09:44');
-- insert into Signups (user_id, time_stamp) values ('6', '2020-12-09 10:39:37');

-- Create table If Not Exists Confirmations (user_id int, time_stamp datetime, action ENUM('confirmed','timeout'));
-- insert into Confirmations (user_id, time_stamp, action) values ('3', '2021-01-06 03:30:46', 'timeout');
-- insert into Confirmations (user_id, time_stamp, action) values ('3', '2021-07-14 14:00:00', 'timeout');
-- insert into Confirmations (user_id, time_stamp, action) values ('7', '2021-06-12 11:57:29', 'confirmed');
-- insert into Confirmations (user_id, time_stamp, action) values ('7', '2021-06-13 12:58:28', 'confirmed');
-- insert into Confirmations (user_id, time_stamp, action) values ('7', '2021-06-14 13:59:27', 'confirmed');
-- insert into Confirmations (user_id, time_stamp, action) values ('2', '2021-01-22 00:00:00', 'confirmed');
-- insert into Confirmations (user_id, time_stamp, action) values ('2', '2021-02-28 23:59:59', 'timeout');


use functions;
-- Drop tables if they already exist
DROP TABLE IF EXISTS orders, products, customers, employees;

-- Customers
CREATE TABLE customers (
    customer_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100),
    email VARCHAR(100),
    city VARCHAR(50),
    join_date DATE
);

-- Employees
CREATE TABLE employees (
    employee_id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100),
    department VARCHAR(50),
    hire_date DATE,
    salary DECIMAL(10, 2)
);

-- Products
CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10, 2),
    stock INT
);

-- Orders
CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    order_date DATE,
    customer_id INT,
    product_id INT,
    employee_id INT,
    quantity INT,
    total_amount DECIMAL(10, 2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
);

-- Insert data into customers
INSERT INTO customers (name, email, city, join_date) VALUES
('Naresh Dangol', 'naresh@example.com', 'Kathmandu', '2023-01-15'),
('John Smith', 'john@example.com', 'New York', '2022-11-20'),
('Sara Lee', 'sara@example.com', 'Chicago', '2023-03-12'),
('Anita Rai', 'anita@example.com', 'Pokhara', '2023-06-10');

-- Insert data into employees
INSERT INTO employees (name, department, hire_date, salary) VALUES
('David Johnson', 'Sales', '2021-05-12', 50000.00),
('Emily Watson', 'Support', '2020-08-30', 48000.00),
('Niraj Karki', 'Sales', '2022-01-10', 52000.00);

-- Insert data into products
INSERT INTO products (product_name, category, price, stock) VALUES
('Laptop', 'Electronics', 1200.00, 25),
('Smartphone', 'Electronics', 800.00, 100),
('Desk Chair', 'Furniture', 150.00, 50),
('Headphones', 'Accessories', 75.00, 150);

-- Insert data into orders
INSERT INTO orders (order_date, customer_id, product_id, employee_id, quantity, total_amount) VALUES
('2024-01-10', 1, 1, 1, 1, 1200.00),
('2024-02-15', 2, 2, 3, 2, 1600.00),
('2024-02-20', 3, 3, 1, 1, 150.00),
('2024-03-01', 1, 4, 2, 2, 150.00),
('2024-03-10', 4, 1, 3, 1, 1200.00),
('2024-03-12', 4, 2, 1, 1, 800.00);
