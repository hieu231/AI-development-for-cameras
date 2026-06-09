#!/usr/bin/env python3
"""Fix ai_models column lengths to match schema definition"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

load_dotenv(override=True)

def fix_column_lengths():
    """Alter ai_models columns to match schema definition"""
    
    # Database connection parameters
    db_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', 5432),
        'database': os.getenv('DB_NAME', 'ai_box_db'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    # Columns that should be varchar(255) according to schema
    columns_to_fix = {
        'name': 255,
        'description': 255,
        'model_path': 255,
        'version': 50  # version can stay at 50
    }
    
    try:
        # Connect to database
        conn = psycopg2.connect(**db_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        print("Checking ai_models column definitions...")
        print("=" * 60)
        
        for column_name, expected_length in columns_to_fix.items():
            # Check current column size
            cursor.execute("""
                SELECT character_maximum_length 
                FROM information_schema.columns 
                WHERE table_name = 'ai_models' 
                AND column_name = %s
            """, (column_name,))
            result = cursor.fetchone()
            
            if result:
                current_length = result[0]
                print(f"\n{column_name}:")
                print(f"  Current: varchar({current_length})")
                print(f"  Expected: varchar({expected_length})")
                
                if current_length != expected_length:
                    print(f"  🔧 Altering to varchar({expected_length})...")
                    cursor.execute(f"""
                        ALTER TABLE ai_models 
                        ALTER COLUMN {column_name} TYPE character varying({expected_length})
                    """)
                    print(f"  ✅ Successfully updated")
                else:
                    print(f"  ✅ Already correct")
            else:
                print(f"\n❌ Column '{column_name}' not found")
        
        print("\n" + "=" * 60)
        print("All column lengths fixed!")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    fix_column_lengths()

