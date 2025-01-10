create database campusx;

use campusx;

CREATE TABLE marks (
 student_id INTEGER PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255),
    branch VARCHAR(255),
    marks INTEGER
);

INSERT INTO marks (name,branch,marks)VALUES 
('Nitish','EEE',82),
('Rishabh','EEE',91),
('Anukant','EEE',69),
('Rupesh','EEE',55),
('Shubham','CSE',78),
('Ved','CSE',43),
('Deepak','CSE',98),
('Arpan','CSE',95),
('Vinay','ECE',95),
('Ankit','ECE',88),
('Anand','ECE',81),
('Rohit','ECE',95),
('Prashant','MECH',75),
('Amit','MECH',69),
('Sunny','MECH',39),
('Gautam','MECH',51);



select * from marks;

select *, avg(marks) over(partition by branch) from marks;

select *,
min(marks) over(partition by branch) as Min,
max(marks) over(partition by branch) as Max,
avg(marks) over(partition by branch) as Avg,
sum(marks) over(partition by branch) as Sum
from marks
order by student_id;


-- All Record Less than AVERAGE
select * from (select *,
			   avg(marks) over(partition by branch) as 'avg'
			   from marks) t
where t.marks > t.avg;										



-- For entire records
select * ,
rank() over(order by marks desc) as RNK 
from marks;

-- For entire records
select * ,
dense_rank() over(order by marks desc) as RNK 
from marks;

select * ,
rank() over(partition by branch order by marks desc) as RNK 
from marks;

-- For entire records
select * ,
dense_rank() over(partition by branch order by marks desc) as RNK 
from marks;


select *,
row_number() over()
from marks;

select *,
row_number() over(partition by branch)
from marks;

select *, concat(t.branch,'-',t.row_num)
from 
	(
	select *,
	row_number() over(partition by branch) as 'row_num'
	from marks
	) as t;
