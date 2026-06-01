from sqlalchemy import Boolean, Column, Integer, String, ForeignKey, DateTime, Text, Float
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime, timezone


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    # Nullable so the column can be added to existing rows by the migration.
    # The /auth/login endpoint refuses login when password_hash is empty.
    password_hash = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    conversations = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )
    agent_jobs = relationship(
        "AgentJob", back_populates="user", cascade="all, delete-orphan"
    )
    github_installations = relationship(
        "GitHubInstallation", back_populates="user", cascade="all, delete-orphan"
    )


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String, default="New Chat")

    # Conversation-scoped chat settings. Set on creation and editable via PATCH.
    system_prompt = Column(Text, default="")
    model = Column(String, nullable=True)
    temperature = Column(Float, default=0.7)
    top_p = Column(Float, default=0.9)
    top_k = Column(Integer, default=40)

    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow)
    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        order_by="Message.id",
        cascade="all, delete-orphan",
    )


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    conversation = relationship("Conversation", back_populates="messages")


class GitHubOAuthServiceConfig(Base):
    __tablename__ = "github_oauth_service_configs"

    id = Column(Integer, primary_key=True, default=1)
    client_id = Column(String, nullable=False)
    client_secret_encrypted = Column(Text, nullable=False)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    updated_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow)


class GitHubOAuthConfig(Base):
    """Legacy per-user OAuth app config kept for schema compatibility."""

    __tablename__ = "github_oauth_configs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    client_id = Column(String, nullable=False)
    client_secret_encrypted = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow)


class GitHubInstallState(Base):
    __tablename__ = "github_install_states"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    state = Column(String, unique=True, nullable=False)
    consumed = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    expires_at = Column(DateTime, nullable=False)


class GitHubInstallation(Base):
    __tablename__ = "github_installations"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    installation_id = Column(String, nullable=False)
    account_login = Column(String, nullable=True)
    account_type = Column(String, nullable=True)
    app_slug = Column(String, nullable=True)
    repository_selection = Column(String, nullable=True)
    permissions_json = Column(Text, default="{}")
    auth_type = Column(String, default="app", nullable=False)
    access_token_encrypted = Column(Text, nullable=True)
    token_scope = Column(Text, nullable=True)
    token_type = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="github_installations")


class AgentJob(Base):
    __tablename__ = "agent_jobs"

    id = Column(String, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="queued", index=True)
    status_detail = Column(Text, nullable=True)

    repository_owner = Column(String, nullable=True)
    repository_name = Column(String, nullable=True)
    base_branch = Column(String, nullable=True)
    target_branch = Column(String, nullable=True)
    commit_sha = Column(String, nullable=True)
    issue_number = Column(Integer, nullable=True)
    pull_request_number = Column(Integer, nullable=True)

    github_installation_id = Column(String, nullable=True)
    github_run_id = Column(String, nullable=True)
    github_check_run_id = Column(String, nullable=True)
    dispatch_event = Column(String, nullable=True)

    metadata_json = Column(Text, nullable=False, default="{}")
    result_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="agent_jobs")
    events = relationship(
        "AgentJobEvent",
        back_populates="job",
        order_by="AgentJobEvent.id",
        cascade="all, delete-orphan",
    )


class AgentJobEvent(Base):
    __tablename__ = "agent_job_events"

    id = Column(Integer, primary_key=True)
    job_id = Column(String, ForeignKey("agent_jobs.id"), nullable=False, index=True)
    status = Column(String, nullable=False)
    event_type = Column(String, nullable=False, default="state")
    message = Column(Text, nullable=True)
    payload_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    job = relationship("AgentJob", back_populates="events")
