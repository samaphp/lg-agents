#!/bin/bash

# Exit on any error
set -e

# Configuration
ECR_REPO="896795400074.dkr.ecr.us-east-1.amazonaws.com/my-ai-agent"
CLUSTER_NAME="agent-cluster"
SERVICE_NAME="my-ai-agent"
REGION="us-east-1"

echo "🚀 Starting deployment process..."

# Login to ECR
echo "📦 Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPO

# Build the Docker image
echo "🔨 Building Docker image..."
docker build  --platform linux/amd64 -t $SERVICE_NAME .

# Tag the image
echo "🏷️ Tagging image..."
docker tag $SERVICE_NAME:latest $ECR_REPO:latest

# Push to ECR
echo "⬆️ Pushing to ECR..."
docker push $ECR_REPO:latest

# Update ECS service
echo "🔄 Updating ECS service..."
aws ecs update-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --force-new-deployment \
    --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:us-east-1:896795400074:targetgroup/my-agent-tg/3502fcffef31dbba,containerName=agent-cluster,containerPort=8123" \
    --region $REGION

echo "⏳ Waiting for deployment to complete..."
aws ecs wait services-stable \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME \
    --region $REGION

echo "✅ Deployment completed successfully!" 

aws elbv2 describe-load-balancers | grep DNSName