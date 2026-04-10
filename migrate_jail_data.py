import os
import json
import logging
from config import ConfigManager
from peewee import *
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("JailMigration")

# Import models from cog
from cogs.jail_cog import JailedUser, HistoricalJailedUser, TomatoCounter, TomatoHistory, UsedMessage, db_proxy

def migrate(config_path: Path):
    logger.info("Starting JailCog data migration...")
    config_manager = ConfigManager(config_path)

    db = config_manager.open_peewee_store("jail_cog.db")
    db_proxy.initialize(db)
    db.connect()
    db.create_tables([JailedUser, HistoricalJailedUser, TomatoCounter, TomatoHistory, UsedMessage])

    server_ids = config_manager.get_all_server_ids()
    cog_id = "JailCog"

    for server_id in server_ids:
        logger.info(f"Looking for data in server {server_id}...")
        data_store_path = config_manager.get_data_store_path(server_id, cog_id)
        
        if not data_store_path.exists():
            logger.info(f"No JailCog data directory found for server {server_id}. Skipping.")
            continue
            
        def get_file(name):
            return data_store_path / name

        def rename_file(old_name, str_ext=".migrated"):
            src = get_file(old_name)
            if src.exists():
                dst = get_file(old_name + str_ext)
                src.rename(dst)
                logger.info(f"Renamed {src.name} to {dst.name}")

        # 1. Migrate currently_jailed_data.json
        curr_jail_file = get_file("currently_jailed_data.json")
        if curr_jail_file.exists():
            try:
                with open(curr_jail_file, "r") as f:
                    curr_data = json.load(f)
                    
                with db.atomic():
                    for uid_str, data in curr_data.items():
                        user_id = int(uid_str)
                        if not JailedUser.select().where((JailedUser.server_id == server_id) & (JailedUser.user_id == user_id)).exists():
                            JailedUser.create(
                                server_id=server_id,
                                user_id=user_id,
                                channel_id=data.get("channel_id", 0),
                                offending_message_id=data.get("offending_message_id", 0),
                                start_time=data.get("start_time", 0),
                                end_time=data.get("end_time", 0),
                                has_been_humiliated=data.get("has_been_humiliated", False)
                            )
                logger.info(f"Migrated currently jailed users for server {server_id}")
                rename_file("currently_jailed_data.json")
                rename_file("currently_jailed_user_ids.json") # We don't migrate this, just rename it so we don't accidentally use it later
            except Exception as e:
                logger.error(f"Failed migrating currently jailed data for {server_id}: {e}")

        # 2. Migrate historical_jailed_data.json
        hist_jail_file = get_file("historical_jailed_data.json")
        if hist_jail_file.exists():
            try:
                with open(hist_jail_file, "r") as f:
                    hist_data = json.load(f)
                    
                with db.atomic():
                    for data in hist_data:
                        HistoricalJailedUser.create(
                            server_id=server_id,
                            user_id=data.get("user_id", 0),
                            channel_id=data.get("channel_id", 0),
                            offending_message_id=data.get("offending_message_id", 0),
                            start_time=data.get("start_time", 0),
                            end_time=data.get("end_time", 0),
                            has_been_humiliated=data.get("has_been_humiliated", False)
                        )
                logger.info(f"Migrated historical jailed users for server {server_id}")
                rename_file("historical_jailed_data.json")
            except Exception as e:
                logger.error(f"Failed migrating historical data for {server_id}: {e}")

        # 3. Migrate used_messages.json
        used_msg_file = get_file("used_messages.json")
        if used_msg_file.exists():
            try:
                with open(used_msg_file, "r") as f:
                    used_msgs = json.load(f)
                    
                with db.atomic():
                    for msg_id in used_msgs:
                        if not UsedMessage.select().where((UsedMessage.server_id == server_id) & (UsedMessage.message_id == msg_id)).exists():
                            UsedMessage.create(server_id=server_id, message_id=msg_id)
                logger.info(f"Migrated used messages for server {server_id}")
                rename_file("used_messages.json")
            except Exception as e:
                logger.error(f"Failed migrating used messages for {server_id}: {e}")

        # 4. Migrate tomato_counters.json
        tomato_cnt_file = get_file("tomato_counters.json")
        if tomato_cnt_file.exists():
            try:
                with open(tomato_cnt_file, "r") as f:
                    tomato_cnts = json.load(f)
                    
                with db.atomic():
                    for uid_str, count in tomato_cnts.items():
                        user_id = int(uid_str)
                        counter, created = TomatoCounter.get_or_create(
                            server_id=server_id,
                            user_id=user_id,
                            defaults={'count': count}
                        )
                        if not created:
                            counter.count += count
                            counter.save()
                logger.info(f"Migrated tomato counters for server {server_id}")
                rename_file("tomato_counters.json")
            except Exception as e:
                logger.error(f"Failed migrating tomato counters for {server_id}: {e}")

        # 5. Migrate tomato_history.json
        tomato_hist_file = get_file("tomato_history.json")
        if tomato_hist_file.exists():
            try:
                with open(tomato_hist_file, "r") as f:
                    tomato_hist = json.load(f)
                    
                with db.atomic():
                    for entry in tomato_hist:
                        # Entry format: (thrower, attacked, msg_id, ts, innocent)
                        if len(entry) == 5:
                            TomatoHistory.create(
                                server_id=server_id,
                                thrower_user_id=entry[0],
                                attacked_user_id=entry[1],
                                message_id=entry[2],
                                timestamp=entry[3],
                                is_innocent=entry[4]
                            )
                logger.info(f"Migrated tomato history for server {server_id}")
                rename_file("tomato_history.json")
            except Exception as e:
                logger.error(f"Failed migrating tomato history for {server_id}: {e}")

    logger.info("Migration complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Migrate JailCog data to Peewee.')
    parser.add_argument('config_path', type=Path, help='Path to the config file', default=Path(__file__).parent / 'configs/config_dev.yaml')
    args = parser.parse_args()
    migrate(args.config_path)
