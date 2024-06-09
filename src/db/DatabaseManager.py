# src/db/DatabaseManager.py

import sqlite3
import logging

class DatabaseManager:
    def __init__(self, db_path='objects.db'):
        self.logger = logging.getLogger('DatabaseManager')
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detected_objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    x INTEGER NOT NULL,
                    y INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL
                )
            ''')
            self.conn.commit()
            self.logger.info("Database table created successfully.")
        except sqlite3.Error as e:
            self.logger.error(f"An error occurred while creating the table: {e}")

    def insert_object(self, label, confidence, x, y, width, height):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO detected_objects (label, confidence, x, y, width, height)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (label, confidence, x, y, width, height))
            self.conn.commit()
            self.logger.info(f"Inserted object: {label} with confidence: {confidence}")
        except sqlite3.Error as e:
            self.logger.error(f"An error occurred while inserting the object: {e}")

    def get_all_objects(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT * FROM detected_objects')
            rows = cursor.fetchall()
            self.logger.info("Fetched all objects from the database.")
            return rows
        except sqlite3.Error as e:
            self.logger.error(f"An error occurred while fetching the objects: {e}")
            return []

    def __del__(self):
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed.")


