import os
from sqlalchemy import create_engine, text
from models import Base, User, UserPreference
from user_service import UserService
from database import DatabaseService
from auth import AuthService

class MigrationService:
    """Service for handling database migrations and data migration"""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL is required for migrations")

        self.engine = create_engine(self.database_url)

    def run_migrations(self) -> bool:
        """Run all necessary migrations"""
        try:
            print("Starting database migrations...")

            # Create new tables
            self._create_tables()

            # Migrate existing data
            self._migrate_existing_data()

            print("Database migrations completed successfully")
            return True

        except Exception as e:
            print(f"Migration failed: {e}")
            return False

    def _create_tables(self):
        """Create all tables (existing tables are not affected)"""
        print("Creating database tables...")
        Base.metadata.create_all(bind=self.engine)
        print("Tables created successfully")

    def _migrate_existing_data(self):
        """Migrate existing conversations to have a default admin user"""
        print("Migrating existing data...")

        try:
            db_service = DatabaseService()
            user_service = UserService(db_service)

            # Ensure default admin user exists
            admin_user = user_service.ensure_default_admin()
            print(f"Default admin user created/verified: {admin_user.username}")

            # Migrate orphaned conversations to admin user
            with db_service.get_session() as session:
                # Find conversations without a user_id
                orphaned_conversations = session.execute(
                    text("SELECT id FROM conversations WHERE user_id IS NULL")
                ).fetchall()

                if orphaned_conversations:
                    print(f"Found {len(orphaned_conversations)} orphaned conversations")

                    # Assign them to the admin user
                    session.execute(
                        text("UPDATE conversations SET user_id = :user_id WHERE user_id IS NULL"),
                        {"user_id": str(admin_user.id)}
                    )

                    print(f"Assigned {len(orphaned_conversations)} conversations to admin user")

            # Migrate file-based preferences to database if they exist
            self._migrate_file_preferences(user_service, admin_user.id)

            print("Data migration completed successfully")

        except Exception as e:
            print(f"Data migration failed: {e}")
            raise

    def _migrate_file_preferences(self, user_service: UserService, admin_user_id: str):
        """Migrate preferences from file to database for admin user"""
        preferences_file = "user_preferences.json"

        if not os.path.exists(preferences_file):
            print("No existing preferences file found, skipping preference migration")
            return

        try:
            import json

            with open(preferences_file, 'r', encoding='utf-8') as f:
                file_preferences = json.load(f)

            if file_preferences:
                print(f"Migrating {len(file_preferences)} preferences from file to database")

                for key, value in file_preferences.items():
                    user_service.set_user_preference(admin_user_id, key, value)

                print("File preferences migrated successfully")

                # Backup the old file
                backup_file = f"{preferences_file}.backup"
                os.rename(preferences_file, backup_file)
                print(f"Old preferences file backed up as {backup_file}")

        except Exception as e:
            print(f"Warning: Could not migrate file preferences: {e}")

    def check_migration_needed(self) -> bool:
        """Check if migration is needed"""
        try:
            with self.engine.connect() as conn:
                # Check if User table exists
                result = conn.execute(
                    text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'users')")
                ).scalar()

                return not result

        except Exception:
            return True

    def backup_database(self, backup_file: str = None) -> bool:
        """Create a backup of the database (PostgreSQL only)"""
        if not backup_file:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"localgpt_backup_{timestamp}.sql"

        try:
            # Extract database info from URL
            from urllib.parse import urlparse
            parsed = urlparse(self.database_url)

            # Run pg_dump
            import subprocess
            cmd = [
                "pg_dump",
                f"--host={parsed.hostname}",
                f"--port={parsed.port or 5432}",
                f"--username={parsed.username}",
                f"--dbname={parsed.path[1:]}",  # Remove leading /
                f"--file={backup_file}",
                "--no-password"
            ]

            # Set password via environment
            env = os.environ.copy()
            env["PGPASSWORD"] = parsed.password

            subprocess.run(cmd, check=True, env=env)
            print(f"Database backup created: {backup_file}")
            return True

        except Exception as e:
            print(f"Backup failed: {e}")
            return False

def run_migrations():
    """Convenience function to run migrations"""
    migration_service = MigrationService()

    if not migration_service.check_migration_needed():
        print("No migration needed - database is up to date")
        return True

    print("Database migration required")

    # Optionally create backup
    backup_choice = input("Create database backup before migration? (y/N): ").lower()
    if backup_choice == 'y':
        if not migration_service.backup_database():
            proceed = input("Backup failed. Continue with migration? (y/N): ").lower()
            if proceed != 'y':
                print("Migration cancelled")
                return False

    return migration_service.run_migrations()

if __name__ == "__main__":
    run_migrations()