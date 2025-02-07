from typing import Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import tweepy

class TwitterPostInput(BaseModel):
    """Input for Twitter post."""
    text: str = Field(..., description="The text content to post on Twitter/X")
    reply_to_tweet_id: Optional[str] = Field(None, description="Optional tweet ID to reply to")

def create_twitter_client():
    """Create and return an authenticated Twitter client."""
    client = tweepy.Client(
        consumer_key="your_api_key",
        consumer_secret="your_api_secret",
        access_token="your_access_token",
        access_token_secret="your_access_token_secret"
    )
    return client

class TwitterPostTool(BaseTool):
    name = "twitter_post"
    description = "Use this tool to post tweets on Twitter/X. Input should be the text you want to tweet."
    args_schema = TwitterPostInput

    def _run(self, text: str, reply_to_tweet_id: Optional[str] = None) -> str:
        client = create_twitter_client()
        
        try:
            if reply_to_tweet_id:
                response = client.create_tweet(
                    text=text,
                    in_reply_to_tweet_id=reply_to_tweet_id
                )
            else:
                response = client.create_tweet(text=text)
            
            tweet_id = response.data['id']
            return f"Successfully posted tweet with ID: {tweet_id}"
        except Exception as e:
            return f"Error posting tweet: {str(e)}"