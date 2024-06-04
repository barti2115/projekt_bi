#!/bin/sh

DB_FILE="/data/offers.db"

if [ ! -f "$DB_FILE" ]; then
    echo "Creating empty database at $DB_FILE..."
    sqlite3 $DB_FILE "VACUUM;"
else
    echo "Database already exists at $DB_FILE"
fi
