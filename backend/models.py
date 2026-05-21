from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
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
    github_installations = relationship(
        "GitHubInstallation", back_populates="user", cascade="all, delete-orphan"
    )
    agent_jobs = relationship(
        "AgentJob", back_populates="user", cascade="all, delete-orphan"
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
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    installation_id = Column(String, nullable=False)
    account_login = Column(String, nullable=True)
    account_type = Column(String, nullable=True)
    app_slug = Column(String, nullable=True)
    repository_selection = Column(String, nullable=True)
    permissions_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow)

    user = relationship("User", back_populates="github_installations")


class AgentJob(Base):
    __tablename__ = "agent_jobs"
    id = Column(String, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String, default="queued", nullable=False)
    repo_full_name = Column(String, nullable=False)
    base_branch = Column(String, nullable=False)
    work_branch = Column(String, nullable=True)
    model = Column(String, nullable=False)
    task = Column(Text, nullable=False)
    test_command = Column(Text, nullable=True)
    push_policy = Column(String, default="direct-main-after-tests", nullable=False)
    commit_sha = Column(String, nullable=True)
    pr_url = Column(String, nullable=True)
    error_summary = Column(Text, nullable=True)
    cancel_requested = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="agent_jobs")
    steps = relationship(
        "AgentJobStep",
        back_populates="job",
        order_by="AgentJobStep.position",
        cascade="all, delete-orphan",
    )
    logs = relationship(
        "AgentJobLog",
        back_populates="job",
        order_by="AgentJobLog.id",
        cascade="all, delete-orphan",
    )
    artifacts = relationship(
        "AgentArtifact",
        back_populates="job",
        order_by="AgentArtifact.id",
        cascade="all, delete-orphan",
    )


class AgentJobStep(Base):
    __tablename__ = "agent_job_steps"
    id = Column(Integer, primary_key=True)
    job_id = Column(String, ForeignKey("agent_jobs.id"), nullable=False)
    position = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    status = Column(String, default="pending", nullable=False)
    exit_code = Column(Integer, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    job = relationship("AgentJob", back_populates="steps")


class AgentJobLog(Base):
    __tablename__ = "agent_job_logs"
    id = Column(Integer, primary_key=True)
    job_id = Column(String, ForeignKey("agent_jobs.id"), nullable=False)
    level = Column(String, default="info", nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utcnow)

    job = relationship("AgentJob", back_populates="logs")


class AgentArtifact(Base):
    __tablename__ = "agent_artifacts"
    id = Column(Integer, primary_key=True)
    job_id = Column(String, ForeignKey("agent_jobs.id"), nullable=False)
    kind = Column(String, nullable=False)
    name = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utcnow)

    job = relationship("AgentJob", back_populates="artifacts")
