import boto3
import json
import time

def setup_aws_infrastructure():
    print("🚀 Starting AWS Infrastructure Setup...")
    
    # Initialize clients
    session = boto3.Session(region_name='us-east-1')
    s3 = session.client('s3')
    rds = session.client('rds')
    elasticache = session.client('elasticache')
    
    # Your unique identifier
    unique_id = input("Enter your initials or unique ID for bucket names: ")
    
    # 1. Create S3 Buckets
    print("📦 Creating S3 buckets...")
    buckets = [
        f"security-system-recordings-{unique_id}",
        f"security-system-backups-{unique_id}"
    ]
    
    for bucket in buckets:
        try:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={'LocationConstraint': 'us-east-1'}
            )
            print(f"✅ Created bucket: {bucket}")
        except Exception as e:
            print(f"⚠️ Bucket {bucket} might already exist: {e}")
    
    # 2. Create RDS Database
    print("🗄️ Creating RDS PostgreSQL database...")
    try:
        db_response = rds.create_db_instance(
            DBName='security_system',
            DBInstanceIdentifier='security-system-db',
            AllocatedStorage=20,
            DBInstanceClass='db.m5d.large',
            Engine='postgres',
            MasterUsername='security_admin',
            MasterUserPassword='security-system-admin987654321',  
            Port=5432,
            PubliclyAccessible=True,
            StorageType='gp2'
        )
        print("✅ RDS database creation started. This will take 10-15 minutes...")
    except Exception as e:
        print(f"⚠️ RDS might already exist: {e}")
    
    # 3. Create Redis Cluster
    print("🔴 Creating Redis cluster...")
    try:
        cache_response = elasticache.create_cache_cluster(
            CacheClusterId='security-system-redis',
            NumCacheNodes=1,
            CacheNodeType='cache.t3.micro',
            Engine='redis',
            EngineVersion='7.0',
            Port=6379
        )
        print("✅ Redis cluster creation started...")
    except Exception as e:
        print(f"⚠️ Redis might already exist: {e}")
    
    print("\n🎉 Setup initiated! Now waiting for resources to be ready...")
    print("⏰ This will take 10-15 minutes for RDS, 5-10 minutes for Redis")
    
    return buckets

if __name__ == "__main__":
    setup_aws_infrastructure()