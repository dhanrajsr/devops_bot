# devops_bot
## DevOps Support ChatBot

``

# Step-by-Step Team Deployment Checklist


### Phase 1: Pre-Deployment Security & Config

• Verify .env Secrets: Confirm OMNI_API_KEY, OMNI_EMBEDDING_API_KEY, DB_USER, DB_PASSWORD, ADMIN_USER, and ADMIN_PASSWORD are set with strong passwords in .env.
• Check .gitignore: Ensure .env, faiss_index/, runbooks_pdf/, and __pycache__/ are listed in .gitignore.
• Optional load_dotenv Safe Mode in app.py: Inside Docker, environment variables are injected directly by docker-compose.yml even without a .env file copied into the image, so load_dotenv(..., override=False) ensures Docker's http://backend:8000 URLs take precedence over any localhost values.

### Phase 2: Build & Launch

• Build and Start All Containers:
      
      docker compose up -d --build

• Verify Container Health: Run docker compose ps and confirm rag-postgres (healthy), rag-fastapi (Up), and rag-streamlit (Up) are all running.
• Inspect Startup Logs: Run docker compose logs -f backend to verify PostgreSQL tables (checkpoints, users, feedback) initialized and the FAISS index loaded cleanly.

### Phase 3: Team Rollout & Hardening

• Test Shared Access: Open http://<SERVER_IP>:8501 from another machine on your corporate network and log in with your ADMIN_USER credentials.
• Create Team Accounts: Use the 👥 Create New User panel in the admin sidebar to provision user-role accounts for your DevOps engineers.
• Reverse Proxy & HTTPS (Recommended): Place Nginx or Caddy in front of port 8501 with your internal TLS certificate so engineers access https://company.com over HTTPS, and restrict port 8000 and 5432 to internal firewall rules only.
• Schedule PostgreSQL Backups: Set up a nightly cron job to back up users, feedback, and chat checkpoints:

       docker exec rag-postgres pg_dump -U rag_user rag_sessions > backup_$(date +%F).sql

``
