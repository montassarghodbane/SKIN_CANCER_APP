CREATE DATABASE skin_cancer_db;
USE skin_cancer_db;

CREATE TABLE Users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR (50),
    password VARCHAR (50)
);

CREATE TABLE patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Name VARCHAR (100),
    age INT,
    result VARCHAR (20),
    probability FLOAT,
    image_path VARCHAR (255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);