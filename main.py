from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form, Depends
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from starlette.middleware.sessions import SessionMiddleware
from utils import (
    process_pdf_safely, extract_durations_optimized, store_chunks_in_cosmos,
    process_batch_with_fallback_accurate, create_excel_with_formatting, generate_document_id,
    task_batches, normalize_and_clean_text, collection, check_existing_chunks,
    test_azure_openai_connection, test_cosmos_connection
)
import os
import logging
import tempfile
from dotenv import load_dotenv
import asyncio
import json
import base64
from typing import Optional
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')]
)
logger = logging.getLogger(__name__)
from datetime import datetime
from utils import update_user_token_stats, create_token_report_excel, get_user_token_summary, log_user_token_usage

# Load environment variables
load_dotenv()

app = FastAPI(title="Project Plan Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple user database (in production, use a proper database)
users = {
    "john doe": {"password": "password123", "name": "John Doe", "role": "Project Manager"},
    "jane smith": {"password": "password123", "name": "Jane Smith", "role": "Business Analyst"},
    "admin": {"password": "admin123", "name": "Administrator", "role": "System Admin"},
}

def get_current_user(request: Request):
    """Get current user from session, raise 401 if not authenticated"""
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve the login page - redirect to main if already logged in"""
    # If user is already logged in with valid session, redirect to main
    user = request.session.get("user")
    if user:
        return RedirectResponse(url="/", status_code=302)
    
    try:
        # Look for login.html in the static directory
        login_path = os.path.join("static", "login.html")
        with open(login_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        # Check if there's an error parameter in the URL
        error = request.query_params.get("error")
        if error:
            html_content = html_content.replace('class="error-message"', 'class="error-message show"')
        
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        logger.error("login.html file not found in static directory")
        # Return a basic login form if file is missing
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>Login</title></head>
        <body>
            <form method="post" action="/login">
                <input type="text" name="username" placeholder="Username" required>
                <input type="password" name="password" placeholder="Password" required>
                <button type="submit">Login</button>
            </form>
        </body>
        </html>
        """, status_code=200)

@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    browser_time: str = Form(None)
):
    """Handle login form submission"""
    logger.info(f"Login attempt for username: {username}")
    
    # Validate credentials
    if username in users and users[username]["password"] == password:
        # Prefer browser time if provided, else fall back to server time
        login_time = browser_time

        request.session["user"] = {
            "username": username,
            "name": users[username]["name"],
            "role": users[username]["role"],
            "login_time": login_time
        }
        
        # Set session expiration (optional - 24 hours)
        request.session["expiry"] = time.time() + 720 * 60 * 60
        
        logger.info(f"Successful login for user: {username} at {login_time}")
        return RedirectResponse(url="/", status_code=303)
    else:
        logger.warning(f"Failed login attempt for username: {username}")
        return RedirectResponse(url="/login?error=1", status_code=303)

@app.get("")
async def root_redirect(request: Request):
    """Redirect root to main landing page"""
    return RedirectResponse(url="/", status_code=302)

# Add a route to clear invalid sessions
@app.get("/clear-session")
async def clear_session(request: Request):
    """Clear any invalid session - useful for development"""
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)

@app.post("/logout")
async def logout(request: Request):
    """Handle logout request"""
    user = request.session.get("user")
    
    if user:
        username = user['username']
        logout_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        login_time = user.get('login_time', 'Unknown')
        
        logger.info(f"User {username} logged out at {logout_time} (logged in at {login_time})")
        
        # Update logout time in user token stats
        try:
            from utils import update_user_logout_time
            update_user_logout_time(username, logout_time)
        except Exception as e:
            logger.error(f"Error updating logout time: {e}")
        
    # Clear session
    request.session.clear()
    return JSONResponse({"status": "success", "message": "Logged out successfully"})

@app.get("/api/user")
async def get_user_info(request: Request):
    """Get current user information"""
    user = get_current_user(request)
    return JSONResponse(user)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serve the main landing page - with proper session validation"""
    # Check if user is logged in with valid session
    user = request.session.get("user")
    
    # If no user in session, redirect to login
    if not user:
        logger.info("No user session found, redirecting to login")
        return RedirectResponse(url="/login", status_code=302)
    
    # Check session expiry if it exists
    expiry = request.session.get("expiry")
    if expiry and time.time() > expiry:
        logger.info(f"Session expired for user: {user.get('username', 'Unknown')}")
        request.session.clear()
        return RedirectResponse(url="/login", status_code=302)
    
    try:
        # Look for index.html in the static directory
        index_path = os.path.join("static", "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        # Replace placeholders with actual user data
        html_content = html_content.replace(
            '<div class="user-avatar">JD</div>',
            f'<div class="user-avatar">{get_initials(user["name"])}</div>'
        )
        html_content = html_content.replace(
            '<span class="user-name">John Doe</span>',
            f'<span class="user-name">{user["name"]}</span>'
        )
        html_content = html_content.replace(
            '<span class="user-role">Project Manager</span>',
            f'<span class="user-role">{user["role"]}</span>'
        )
        
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        logger.error("static/index.html file not found")
        raise HTTPException(status_code=500, detail="Dashboard not found")
    except Exception as e:
        logger.error(f"Error serving frontend: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load dashboard")
    
def get_initials(name: str) -> str:
    """Get initials from a full name"""
    if not name:
        return "UN"
    
    parts = name.strip().split()
    if len(parts) >= 2:
        return f"{parts[0][0]}{parts[1][0]}".upper()
    elif len(parts) == 1:
        return parts[0][:2].upper()
    else:
        return "UN"

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    request: Request = None
):
    """OPTIMIZED: Handle PDF upload with TPM fixes"""
    user = get_current_user(request)
    logger.info(f"OPTIMIZED: File upload from user: {user['username']} for file: {file.filename}")
    
    total_processing_start_time = time.time()
    total_ai_time = 0
    
    try:
        # Validation
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(file_content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 20MB)")
        
        await file.seek(0)
        
        # Test Azure services
        try:
            from utils import llm
            test_response = llm.invoke("Say 'connection test' only.")
            logger.info("Azure OpenAI connectivity verified")
        except Exception as e:
            raise HTTPException(status_code=503, detail="AI service unavailable.")

        # Create a copy of current token stats BEFORE processing
        from utils import token_stats
        initial_token_stats = token_stats.copy()
        logger.info(f"Initial token stats: {initial_token_stats}")
        
        loop = asyncio.get_event_loop()
        
        # Step 1: Extract PDF content
        logger.info("Step 1: Extracting PDF content...")
        processing_result = await loop.run_in_executor(None, lambda: process_pdf_safely(file))
        if processing_result is None:
            raise HTTPException(status_code=400, detail="No readable content found in the PDF")

        pdf_text, normalized_pdf_text, tmp_pdf_path, images_content = processing_result
        
        # Step 2: Extract durations
        logger.info("Step 2: Extracting phase durations...")
        durations, durations_ai_time = await loop.run_in_executor(None, lambda: extract_durations_optimized(pdf_text))
        total_ai_time += durations_ai_time

        # Step 3: Process tasks with OPTIMIZED function
        logger.info("Step 3: Analyzing tasks with optimized detection...")
        all_tasks = [(str(heading), str(task)) for heading, tasks in task_batches.items() for task in tasks if task and task.strip()]
        if not all_tasks:
            raise HTTPException(status_code=400, detail="No valid tasks found to process")

        # Use smaller batch size for better TPM management
        batch_size = 15  # Reduced from 25
        task_batches_split = [all_tasks[i:i + batch_size] for i in range(0, len(all_tasks), batch_size)]

        results = []

        # Process batches with OPTIMIZED function and delays
        for idx, batch in enumerate(task_batches_split):
            logger.info(f"Processing batch {idx + 1} of {len(task_batches_split)}")
            
            # Add delay between batches to manage TPM
            if idx > 0:
                await asyncio.sleep(3)  # 3 second delay between batches
            
            result, batch_ai_time = await loop.run_in_executor(
                None, 
                lambda: process_batch_with_fallback_accurate(
                    batch, durations, normalized_pdf_text, pdf_text
                )
            )
            total_ai_time += batch_ai_time
            if result:
                results.append(result)

        flat_rows = [row for result in results for row in result if result and isinstance(result, list)]
        if not flat_rows:
            raise HTTPException(status_code=500, detail="Failed to process any tasks")

        import pandas as pd
        df = pd.DataFrame(flat_rows)
        df = df[df['Present'] != 'error']
        if df.empty:
            raise HTTPException(status_code=500, detail="All tasks failed processing")

        # Count tasks per phase
        phase_task_counts = count_tasks_per_phase(df)
        
        # Create Excel file
        logger.info("Creating Excel report...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_excel:
            await loop.run_in_executor(None, lambda: create_excel_with_formatting(df, durations, tmp_excel.name, activity_column_width=50))
            tmp_excel_path = tmp_excel.name

        # Cleanup
        if tmp_pdf_path and os.path.exists(tmp_pdf_path):
            os.unlink(tmp_pdf_path)

        # Calculate processing time
        total_processing_time = time.time() - total_processing_start_time
        
        # Format times
        if total_processing_time >= 60:
            minutes = int(total_processing_time // 60)
            seconds = int(total_processing_time % 60)
            formatted_total_time = f"{minutes} min {seconds} sec"
        else:
            formatted_total_time = f"{total_processing_time:.1f} sec"
        
        # Calculate actual token usage during this session
        current_token_stats = token_stats.copy()
        session_token_stats = {
            "llm_input_tokens": current_token_stats["llm_input_tokens"] - initial_token_stats["llm_input_tokens"],
            "llm_output_tokens": current_token_stats["llm_output_tokens"] - initial_token_stats["llm_output_tokens"],
            "embedding_tokens": current_token_stats["embedding_tokens"] - initial_token_stats["embedding_tokens"],
            "llm_calls": current_token_stats["llm_calls"] - initial_token_stats["llm_calls"],
            "embedding_calls": current_token_stats["embedding_calls"] - initial_token_stats["embedding_calls"],
        }
        
        logger.info(f"Session token usage: {session_token_stats}")
        
        # Update user token stats with the session data
        login_time = user.get('login_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        update_user_token_stats(
            session_token_stats,
            file.filename,
            user['username'],
            user.get('email', 'Unknown'),
            login_time,
            "Not logged out",
            total_ai_time,
            total_processing_time
        )
        
        # Prepare metadata
        completed_tasks = len(df[df['Present'] == 'yes'])
        total_tasks = len(df)
        unique_headings = df['Heading'].nunique()
        
        metadata = {
            "durations": durations,
            "completedTasks": completed_tasks,
            "totalTasks": total_tasks,
            "totalPhases": unique_headings,
            "phasesWithDurations": len([d for d in durations.values() if d and str(d).strip()]),
            "uniqueHeadings": unique_headings,
            "phaseTaskCounts": phase_task_counts,
            "totalProcessingTime": formatted_total_time,
            "aiResponseTime": f"{total_ai_time:.1f} sec"
        }

        # Read the Excel file
        with open(tmp_excel_path, 'rb') as f:
            excel_data = base64.b64encode(f.read()).decode('utf-8')

        # Cleanup
        if os.path.exists(tmp_excel_path):
            os.unlink(tmp_excel_path)

        logger.info(f"SUCCESS: Optimized processing completed in {formatted_total_time}")
        logger.info(f"Task distribution: {completed_tasks}/{total_tasks} completed tasks")
        return JSONResponse({
            "status": "success",
            "metadata": metadata,
            "file": excel_data,
            "filename": "AI-Generated_SOW_Document.xlsx"
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
def count_tasks_per_phase(df):
    """Count tasks per phase dynamically from the DataFrame"""
    phase_counts = {}
    
    # Group by Heading (phase) and count tasks
    for heading in df['Heading'].unique():
        phase_tasks = df[df['Heading'] == heading]
        phase_counts[heading] = {
            'total_tasks': len(phase_tasks),
            'completed_tasks': len(phase_tasks[phase_tasks['Present'] == 'yes'])
        }
    
    return phase_counts

@app.get("/debug/config")
async def debug_config(request: Request):
    """Comprehensive debug endpoint to check configuration"""
    try:
        user = get_current_user(request)
    except:
        user = {"username": "anonymous", "role": "unknown"}
    
    config_status = {
        "service": "SOW Analyzer",
        "user": user['username'],
        "environment": "Azure",
        "timestamp": datetime.now().isoformat()
    }
    
    # Check environment variables (without exposing secrets)
    env_vars = {
        "AZURE_OPENAI_ENDPOINT": bool(os.getenv("AZURE_OPENAI_ENDPOINT")),
        "AZURE_OPENAI_API_KEY": "***" if os.getenv("AZURE_OPENAI_API_KEY") else None,
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
        "AZURE_OPENAI_CHAT_DEPLOYMENT": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        "COSMOS_URI": "***" if os.getenv("COSMOS_URI") else None,
        "COSMOS_DB": os.getenv("COSMOS_DB"),
        "COSMOS_COLLECTION": os.getenv("COSMOS_COLLECTION"),
    }
    config_status["environment_variables"] = env_vars
    
    # Test Azure OpenAI
    try:
        openai_test = test_azure_openai_connection()
        config_status["azure_openai"] = openai_test
    except Exception as e:
        config_status["azure_openai"] = {
            "status": "failed",
            "error": str(e)
        }
    
    # Test Cosmos DB
    try:
        cosmos_test = test_cosmos_connection()
        config_status["cosmos_db"] = cosmos_test
    except Exception as e:
        config_status["cosmos_db"] = {
            "status": "failed",
            "error": str(e)
        }
    
    return JSONResponse(config_status)

@app.get("/debug/token-stats")
async def debug_token_stats(request: Request):
    """Debug endpoint to check token statistics"""
    user = get_current_user(request)
    
    from utils import user_token_stats, token_stats
    
    debug_info = {
        "current_user": user['username'],
        "global_token_stats": token_stats,
        "user_token_stats_keys": list(user_token_stats.keys()) if user_token_stats else [],
        "current_user_in_stats": user['username'] in user_token_stats if user_token_stats else False,
    }
    
    if user['username'] in user_token_stats:
        debug_info["user_stats"] = user_token_stats[user['username']]
    
    return JSONResponse(debug_info)

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Project Plan Agent",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/dashboard")
async def dashboard_page(request: Request):
    """Serve the dashboard page with token reports"""
    user = get_current_user(request)
    try:
        # Get token summary data for the table
        token_summary = get_user_token_summary()

        # Prepare admin button
        admin_button = "<a href='/download-token-report' class='btn btn-admin'>📋 Download All Users Report (Admin)</a>" if user['role'] == 'System Admin' else ""

        # Dashboard HTML
        dashboard_html = """
         <!DOCTYPE html>
         <html>
         <head>
             <title>Dashboard - Token Reports</title>
             <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                 .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                 .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; padding-bottom: 12px; border-bottom: 1px solid #e0e0e0; }
                 .user-info { text-align: right; font-size: 13px; }
                 /* Hamburger styling for dashboard header */
                 .hamburger { width: 36px; height: 36px; display: inline-flex; align-items: center; justify-content: center; cursor: pointer; border-radius: 8px; transition: background .15s ease; }
                 .hamburger:hover { background: rgba(0,0,0,0.04); }
                 .hamburger-lines { position: relative; width: 20px; height: 14px; }
                 .hamburger-lines span { position: absolute; left: 0; right: 0; height: 2px; background: #333; display: block; border-radius: 2px; transition: transform .18s ease, opacity .18s ease; }
                 .hamburger-lines span:nth-child(1) { top: 0; }
                 .hamburger-lines span:nth-child(2) { top: 6px; }
                 .hamburger-lines span:nth-child(3) { top: 12px; }
                 .hamburger.active .hamburger-lines span:nth-child(1) { transform: translateY(6px) rotate(45deg); }
                 .hamburger.active .hamburger-lines span:nth-child(2) { opacity: 0; transform: scaleX(0); }
                 .hamburger.active .hamburger-lines span:nth-child(3) { transform: translateY(-6px) rotate(-45deg); }
                 .card { background: #f8f9fa; padding: 18px; border-radius: 8px; margin-bottom: 18px; border-left: 4px solid #007bff; }
                 .btn { background: #007bff; color: white; padding: 8px 14px; text-decoration: none; border-radius: 6px; display: inline-block; margin-right: 8px; margin-bottom: 8px; border: none; cursor: pointer; font-size: 13px; }
                 .btn:hover { background: #0056b3; transform: translateY(-1px); }
                 .btn-admin { background: #28a745; }
                 .btn-back { background: #6c757d; margin-right: 10px; }
                 .btn-back:hover { background: #5a6268; }
                 table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }
                 th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                 th { background: #007bff; color: white; font-weight: bold; }
                 .stats-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 18px; }
                 .stat-card { background: white; padding: 12px; border-radius: 8px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.06); border-top: 4px solid #007bff; }
                 .stat-number { font-size: 1.25rem; font-weight: bold; color: #007bff; }
                 .stat-label { color: #666; margin-top: 6px; font-size: 12px; }
                 .error { color: #dc3545; background: #f8d7da; padding: 8px; border-radius: 5px; margin: 10px 0; font-size: 13px; }
                 .success { color: #155724; background: #d4edda; padding: 8px; border-radius: 5px; margin: 10px 0; font-size: 13px; }
                 .header-left { display: flex; align-items: center; gap: 12px; }
                 .page-title { margin: 0; font-size: 18px; }
                 .admin-badge { background: #28a745; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; margin-left: 8px; }
                 .user-only-stats { border-top: 4px solid #28a745; }
             </style>
         </head>
         <body>
             <div class="container">
                 <div class="header">
                     <div class="header-left">
                         <div class="hamburger" id="hamburger"><div class="hamburger-lines"><span></span><span></span><span></span></div></div>
                         <h1 class="page-title">📊 Token Usage Dashboard {ADMIN_BADGE}</h1>
                     </div>
                     <div class="user-info">
                         <strong>👤 Welcome, {USER_NAME}</strong><br/>
                         <span>🏷️ {USER_ROLE}</span><br/>
                         <span>📧 {USER_USERNAME}</span>
                     </div>
                 </div>
                 
                 <!-- Back Button -->
                 <div style="margin-bottom: 18px;">
                     <a href="/back-to-main" class="btn btn-back">⬅ Back to Main Page</a>
                 </div>
                 
                 <!-- Quick Stats - ALWAYS show AGGREGATED stats across all users -->
                 <div class="stats-container" id="quickStats">
                     <div class="stat-card">
                         <div class="stat-number" id="totalUsers">0</div>
                         <div class="stat-label">Total Users</div>
                     </div>
                     <div class="stat-card">
                         <div class="stat-number" id="totalFiles">0</div>
                         <div class="stat-label">Files Processed</div>
                     </div>
                     <div class="stat-card">
                         <div class="stat-number" id="totalTokens">0</div>
                         <div class="stat-label">Total Tokens</div>
                     </div>
                     <div class="stat-card">
                         <div class="stat-number" id="avgTokens">0</div>
                         <div class="stat-label">Avg Tokens/File</div>
                     </div>
                 </div>
                 
                 <div class="card">
                     <h2 style="margin-top:0;">📥 Download Token Reports</h2>
                     <p style="margin:6px 0 10px 0;">Download comprehensive token usage reports for analysis and monitoring.</p>
                     
                     <!-- Error/Success Messages -->
                     <div id="messageContainer"></div>
                     
                     <div style="margin-top: 12px;">
                         {ADMIN_BUTTON}
                         <a href='/download-my-token-report' class='btn'>👤 Download My Usage Report</a>
                     </div>
                 </div>
                 
                 <div class="card">
                     <h2 style="margin-top:0;" id="analyticsTitle">📈 Your Token Usage Analytics</h2>
                     <p style="margin:6px 0 10px 0;" id="analyticsDescription">Detailed view of your personal token usage and session history.</p>
                     <div id="tokenSummary">
                         <p>Loading token usage data...</p>
                     </div>
                 </div>
             </div>
             
             <script>
                // Load token summary data
                async function loadTokenSummary() {
                    try {
                        const response = await fetch('/api/token-sessions');
                        if (response.ok) {
                            const data = await response.json();
                            displayTokenSummary(data);
                            updateQuickStats(data);
                        } else {
                            showMessage('Error loading token data: ' + response.status, 'error');
                        }
                    } catch (error) {
                        console.error('Error loading token summary:', error);
                        showMessage('Network error loading token data', 'error');
                    }
                }
                
                function updateQuickStats(data) {
                    // Calculate AGGREGATED stats across all users
                    if (data.aggregated_stats) {
                        document.getElementById('totalUsers').textContent = data.aggregated_stats.total_users.toLocaleString();
                        document.getElementById('totalFiles').textContent = data.aggregated_stats.total_files.toLocaleString();
                        document.getElementById('totalTokens').textContent = data.aggregated_stats.total_tokens.toLocaleString();
                        document.getElementById('avgTokens').textContent = data.aggregated_stats.avg_tokens_per_file.toLocaleString();
                    } else {
                        // No data available
                        document.getElementById('totalUsers').textContent = '0';
                        document.getElementById('totalFiles').textContent = '0';
                        document.getElementById('totalTokens').textContent = '0';
                        document.getElementById('avgTokens').textContent = '0';
                    }
                }
                
                function displayTokenSummary(data) {
                    const summaryDiv = document.getElementById('tokenSummary');
                    
                    // Get user role to determine display logic
                    const userRole = data.user_role;
                    
                    if (data.user_summary && Object.keys(data.user_summary).length > 0) {
                        let html = '<table>';
                        
                        // Adjust table headers based on user role
                        if (userRole === 'System Admin') {
                            html += '<tr><th>Username</th><th>File Name</th><th>Total Tokens</th><th>Total Processing Time</th><th>User Login Time</th></tr>';
                        } else {
                            html += '<tr><th>Username</th><th>File Name</th><th>Total Tokens</th><th>Total Processing Time</th><th>User Login Time</th></tr>'; // ADDED Username column
                        }
                        
                        // Show ALL sessions
                        for (const [sessionKey, stats] of Object.entries(data.user_summary)) {
                            if (userRole === 'System Admin') {
                                html += `<tr>
                                    <td><strong>${stats.username || 'Unknown'}</strong></td>
                                    <td>${stats.pdf_filename || 'Unknown'}</td>
                                    <td>${(stats.total_tokens || 0).toLocaleString()}</td>
                                    <td>${stats.total_response_time || 'N/A'}</td>
                                    <td>${stats.last_activity || 'Unknown'}</td>
                                </tr>`;
                            } else {
                                html += `<tr>
                                    <td><strong>${stats.username || 'Unknown'}</strong></td> <!-- ADDED Username cell -->
                                    <td>${stats.pdf_filename || 'Unknown'}</td>
                                    <td>${(stats.total_tokens || 0).toLocaleString()}</td>
                                    <td>${stats.total_response_time || 'N/A'}</td>
                                    <td>${stats.last_activity || 'Unknown'}</td>
                                </tr>`;
                            }
                        }
                        
                        html += '</table>';
                        summaryDiv.innerHTML = html;
                    } else {
                        summaryDiv.innerHTML = '<p class="error">No token data available. Process some PDF files to see usage statistics.</p>';
                    }
                }
                
                function showMessage(message, type) {
                    const messageContainer = document.getElementById('messageContainer');
                    messageContainer.innerHTML = `<div class="${type === 'error' ? 'error' : 'success'}">${message}</div>`;
                    setTimeout(() => messageContainer.innerHTML = '', 5000);
                }

                document.addEventListener('DOMContentLoaded', function() {
                     const downloadLinks = document.querySelectorAll('a[href*="download-token-report"]');
                     
                     downloadLinks.forEach(link => {
                         link.addEventListener('click', async function(e) {
                             e.preventDefault();
                             const href = this.href;
                             
                             try {
                                 const response = await fetch(href);
                                 if (response.ok) {
                                     const blob = await response.blob();
                                     const url = window.URL.createObjectURL(blob);
                                     const a = document.createElement('a');
                                     a.href = url;
                                     a.download = this.textContent.includes('All Users') ? 'token_report_all_users.xlsx' : 'token_report_my_usage.xlsx';
                                     document.body.appendChild(a);
                                     a.click();
                                     window.URL.revokeObjectURL(url);
                                     document.body.removeChild(a);
                                     showMessage('Download started successfully!', 'success');
                                 } else {
                                     const errorData = await response.json();
                                     showMessage('Download failed: ' + (errorData.detail || 'Unknown error'), 'error');
                                 }
                             } catch (error) {
                                 showMessage('Network error: ' + error.message, 'error');
                             }
                         });
                     });
                 });

                // Hamburger toggle handler
                document.getElementById('hamburger').addEventListener('click', function() {
                    this.classList.toggle('active');
                });
                
                // Load summary when page loads
                loadTokenSummary();
             </script>
         </body>
         </html>
         """

        # Inject dynamic values and admin badge
        admin_badge = "<span class='admin-badge'>ADMIN</span>" if user['role'] == 'System Admin' else ""
        
        dashboard_html = dashboard_html.replace("{USER_NAME}", user.get('name', 'User')) \
                                       .replace("{USER_ROLE}", user.get('role', '')) \
                                       .replace("{USER_USERNAME}", user.get('username', '')) \
                                       .replace("{ADMIN_BUTTON}", admin_button) \
                                       .replace("{ADMIN_BADGE}", admin_badge)

        return HTMLResponse(content=dashboard_html)

    except Exception as e:
        logger.error(f"Error serving dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load dashboard")
                    
@app.get("/download-token-report")
async def download_token_report(request: Request):
    """Download token report for all users (admin only)"""
    user = get_current_user(request)
    
    # Check if user is admin
    if user['role'] != 'System Admin':
        raise HTTPException(status_code=403, detail="Access denied. Admin privileges required.")
    
    try:
        # Create the token report
        token_report = create_token_report_excel()  # All users report
        if token_report:
            # Create a temporary file to serve
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                tmp_file.write(token_report.getvalue())
                tmp_file_path = tmp_file.name
            
            # Return file response with background cleanup
            return FileResponse(
                tmp_file_path,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename="token_report_all_users.xlsx",
                background=BackgroundTask(lambda: os.unlink(tmp_file_path) if os.path.exists(tmp_file_path) else None)
            )
        else:
            raise HTTPException(status_code=404, detail="No token data available")
    except Exception as e:
        logger.error(f"Error generating token report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate token report: {str(e)}")

@app.get("/download-my-token-report")
async def download_my_token_report(request: Request):
    """Download token report for current user"""
    user = get_current_user(request)
    
    try:
        # Create the token report for current user
        token_report = create_token_report_excel(user['username'])
        if token_report:
            # Create a temporary file to serve
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                tmp_file.write(token_report.getvalue())
                tmp_file_path = tmp_file.name
            
            # Return file response with background cleanup
            return FileResponse(
                tmp_file_path,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename=f"token_report_{user['username']}.xlsx",
                background=BackgroundTask(lambda: os.unlink(tmp_file_path) if os.path.exists(tmp_file_path) else None)
            )
        else:
            raise HTTPException(status_code=404, detail="No token data available for your account")
    except Exception as e:
        logger.error(f"Error generating user token report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate token report: {str(e)}")
    
@app.get("/api/token-summary")
async def get_token_summary(request: Request):
    """Get token usage summary for dashboard"""
    user = get_current_user(request)
    
    try:
        summary = get_user_token_summary()
        
        # For regular users: return only their data in both summary and user_summary
        if user['role'] == 'System Admin':
            user_summary = summary  # Admin sees all users data
        else:
            # Regular users see only their data in BOTH summary and user_summary
            user_summary = {}
            summary = {}  # Clear the all-users summary for regular users
            if user['username'] in get_user_token_summary():
                user_data = get_user_token_summary()[user['username']]
                user_summary[user['username']] = user_data
                summary[user['username']] = user_data  # Also put in summary for consistency
        
        return JSONResponse({
            "summary": summary,  # User-specific data for regular users, all users for admin
            "user_summary": user_summary,  # User-specific data for regular users, all users for admin
            "user_role": user['role']
        })
    except Exception as e:
        logger.error(f"Error getting token summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get token summary")
    
@app.get("/api/token-sessions")
async def get_token_sessions(request: Request):
    """Get token usage session data for dashboard - returns ALL session records"""
    user = get_current_user(request)
    
    try:
        from utils import user_token_stats
        
        # Calculate aggregated statistics across ALL users
        total_users = len(user_token_stats)
        total_files = 0
        total_tokens = 0
        
        for username, user_stats in user_token_stats.items():
            total_files += user_stats.get('total_files_processed', 0)
            total_tokens += (user_stats.get('llm_input_tokens', 0) + 
                           user_stats.get('llm_output_tokens', 0) + 
                           user_stats.get('embedding_tokens', 0))
        
        avg_tokens_per_file = round(total_tokens / total_files) if total_files > 0 else 0
        
        aggregated_stats = {
            'total_users': total_users,
            'total_files': total_files,
            'total_tokens': total_tokens,
            'avg_tokens_per_file': avg_tokens_per_file
        }
        
        # Get ALL sessions for the appropriate user scope
        all_sessions = []
        if user['role'] == 'System Admin':
            # Admin gets ALL sessions from ALL users
            for username, user_stats in user_token_stats.items():
                user_sessions = user_stats.get('processing_sessions', [])
                for session in user_sessions:
                    all_sessions.append(session)
        else:
            # Regular user gets ALL their own sessions
            if user['username'] in user_token_stats:
                all_sessions = user_token_stats[user['username']].get('processing_sessions', [])
        
        # Sort sessions by timestamp (newest first)
        all_sessions.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        # Format the session data for display - ALL sessions
        formatted_sessions = []
        for session in all_sessions:
            formatted_sessions.append({
                'username': session.get('username', 'Unknown'),
                'pdf_filename': session.get('pdf_filename', 'Unknown'),
                'total_tokens': session.get('total_tokens', 0),
                'formatted_total_time': session.get('formatted_total_time', 'N/A'),
                'login_time': session.get('login_time', 'Unknown'),
                'logout_time': session.get('logout_time', 'Not logged out'),
                'llm_calls': session.get('llm_calls', 0),
                'llm_input_tokens': session.get('llm_input_tokens', 0),
                'llm_output_tokens': session.get('llm_output_tokens', 0),
                'embedding_tokens': session.get('embedding_tokens', 0),
                'timestamp': session.get('timestamp', 0)
            })
        
        # Prepare user summary data for the detailed table - show ALL sessions
        user_summary = {}
        if user['role'] == 'System Admin':
            # Admin sees ALL sessions from ALL users
            for username, user_stats in user_token_stats.items():
                sessions = user_stats.get('processing_sessions', [])
                for session in sessions:
                    # Create a unique key for each session to show all rows
                    session_key = f"{username}_{session.get('timestamp', '')}"
                    user_summary[session_key] = {
                        'pdf_filename': session.get('pdf_filename', 'Unknown'),
                        'total_tokens': session.get('total_tokens', 0),
                        'total_response_time': session.get('formatted_total_time', 'N/A'),
                        'last_activity': session.get('login_time', 'Unknown'),
                        'username': username  # Include username for admin view
                    }
        else:
            # Regular user sees ALL their own sessions
            if user['username'] in user_token_stats:
                sessions = user_token_stats[user['username']].get('processing_sessions', [])
                for session in sessions:
                    # Create a unique key for each session to show all rows
                    session_key = f"{user['username']}_{session.get('timestamp', '')}"
                    user_summary[session_key] = {
                        'pdf_filename': session.get('pdf_filename', 'Unknown'),
                        'total_tokens': session.get('total_tokens', 0),
                        'total_response_time': session.get('formatted_total_time', 'N/A'),
                        'last_activity': session.get('login_time', 'Unknown'),
                        'username': user['username']
                    }
        
        return JSONResponse({
            "sessions": formatted_sessions,
            "user_summary": user_summary,
            "aggregated_stats": aggregated_stats,
            "user_role": user['role']
        })
        
    except Exception as e:
        logger.error(f"Error getting token sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get token session data")
            
@app.get("/back-to-main")
async def back_to_main(request: Request):
    """Redirect back to main landing page"""
    user = get_current_user(request)
    logger.info(f"User {user['username']} navigating back to main page")
    return RedirectResponse(url="/", status_code=302)

# Add middleware after route definitions to ensure proper ordering
@app.middleware("http")
async def check_session_validity(request: Request, call_next):
    """Middleware to check session validity"""
    # Skip session check for login page and static files
    if request.url.path in ["/login", "/static", "/health", "/debug/config", "/debug/token-stats"] or request.url.path.startswith("/static/"):
        return await call_next(request)
    
    # Check if session exists and has user data
    user = request.session.get("user")
    if not user:
        # If no user session and trying to access protected routes, redirect to login
        if request.url.path not in ["/login", "/static", "/clear-session"] and not request.url.path.startswith("/static/"):
            logger.info(f"Redirecting to login from {request.url.path} - no session")
            return RedirectResponse(url="/login", status_code=302)
    
    # Check session expiry if it exists
    expiry = request.session.get("expiry")
    if expiry and time.time() > expiry:
        logger.info(f"Session expired for user: {user.get('username', 'Unknown') if user else 'Unknown'}")
        request.session.clear()
        if request.url.path not in ["/login", "/static"] and not request.url.path.startswith("/static/"):
            return RedirectResponse(url="/login", status_code=302)
    
    response = await call_next(request)
    return response

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add session middleware with secret key
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "your-secret-key-change-in-production")
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
