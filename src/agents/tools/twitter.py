from typing import Optional
from pydantic import BaseModel, Field
import tweepy

class TwitterPostConfig(BaseModel):
    """Configuration for Twitter post."""
    text: str = Field(..., description="The text content to post on Twitter/X")
    reply_to_tweet_id: Optional[str] = Field(None, description="Optional tweet ID to reply to")
    consumer_key: str = Field(..., description="The consumer key for the Twitter API")
    consumer_secret: str = Field(..., description="The consumer secret for the Twitter API")
    access_token: str = Field(..., description="The access token for the Twitter API")
    access_token_secret: str = Field(..., description="The access token secret for the Twitter API")

def create_twitter_client(consumer_key: str, consumer_secret: str, access_token: str, access_token_secret: str) -> tweepy.Client:
    """Create and return an authenticated Twitter client."""
    return tweepy.Client(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_token_secret
    )

def post_tweet(config: TwitterPostConfig) -> str:
    """
    Post a tweet to Twitter/X.
    
    Args:
        config: TwitterPostConfig containing all necessary credentials and tweet content
        
    Returns:
        str: Success message with tweet ID or error message
    """
    client = create_twitter_client(
        consumer_key=config.consumer_key,
        consumer_secret=config.consumer_secret,
        access_token=config.access_token,
        access_token_secret=config.access_token_secret
    )
    
    try:
        if config.reply_to_tweet_id:
            response = client.create_tweet(
                text=config.text,
                in_reply_to_tweet_id=config.reply_to_tweet_id
            )
        else:
            response = client.create_tweet(text=config.text)
        
        tweet_id = response.data['id']
        return f"Successfully posted tweet with ID: {tweet_id}"
    except Exception as e:
        return f"Error posting tweet: {str(e)}"