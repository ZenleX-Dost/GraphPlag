-- PostgreSQL initialization script for GraphPlag scalable system

-- Create extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    doc_id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(1024),
    file_size BIGINT,
    file_hash VARCHAR(64) UNIQUE,
    content_type VARCHAR(50),
    content TEXT,
    added_to_corpus BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analyses table - stores plagiarism analysis results
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id SERIAL PRIMARY KEY,
    job_id UUID UNIQUE NOT NULL,
    doc_id INTEGER REFERENCES documents(doc_id),
    file_name VARCHAR(255) NOT NULL,
    ai_score FLOAT DEFAULT 0.0,
    num_matches INTEGER DEFAULT 0,
    total_processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Matches table - individual plagiarism matches
CREATE TABLE IF NOT EXISTS matches (
    match_id SERIAL PRIMARY KEY,
    analysis_id INTEGER REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    job_id UUID NOT NULL REFERENCES analyses(job_id),
    rank INTEGER,
    matched_doc_id INTEGER REFERENCES documents(doc_id),
    matched_file_name VARCHAR(255),
    vector_similarity_score FLOAT DEFAULT 0.0,
    fulltext_similarity_score FLOAT DEFAULT 0.0,
    combined_similarity_score FLOAT DEFAULT 0.0,
    matched_ai_score FLOAT DEFAULT 0.0,
    plagiarism_percentage FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document embeddings metadata
CREATE TABLE IF NOT EXISTS document_embeddings (
    embedding_id SERIAL PRIMARY KEY,
    doc_id INTEGER UNIQUE REFERENCES documents(doc_id) ON DELETE CASCADE,
    embedding_model VARCHAR(100),
    embedding_dim INTEGER,
    milvus_id BIGINT UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Job status tracking
CREATE TABLE IF NOT EXISTS job_status (
    job_id UUID PRIMARY KEY,
    status VARCHAR(50) DEFAULT 'queued',
    progress INTEGER DEFAULT 0,
    current_step VARCHAR(255),
    message TEXT,
    eta_seconds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analysis results (for streaming)
CREATE TABLE IF NOT EXISTS analysis_results (
    result_id SERIAL PRIMARY KEY,
    job_id UUID UNIQUE REFERENCES job_status(job_id),
    data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Batch processing jobs (Spark)
CREATE TABLE IF NOT EXISTS batch_jobs (
    batch_job_id SERIAL PRIMARY KEY,
    job_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    input_documents INTEGER,
    processed_documents INTEGER,
    spark_job_id VARCHAR(255),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics
CREATE TABLE IF NOT EXISTS metrics (
    metric_id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT,
    tags JSONB,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System logs
CREATE TABLE IF NOT EXISTS logs (
    log_id SERIAL PRIMARY KEY,
    level VARCHAR(20),
    logger VARCHAR(100),
    message TEXT,
    job_id UUID,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common queries

-- Documents table indexes
CREATE INDEX IF NOT EXISTS idx_file_name ON documents(file_name);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);

-- Analyses table indexes
CREATE INDEX IF NOT EXISTS idx_job_id_analyses ON analyses(job_id);
CREATE INDEX IF NOT EXISTS idx_doc_id_analyses ON analyses(doc_id);
CREATE INDEX IF NOT EXISTS idx_created_at_analyses ON analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_analyses_ai_score ON analyses(ai_score);

-- Matches table indexes  
CREATE INDEX IF NOT EXISTS idx_analysis_id_matches ON matches(analysis_id);
CREATE INDEX IF NOT EXISTS idx_job_id_matches ON matches(job_id);
CREATE INDEX IF NOT EXISTS idx_rank ON matches(rank);
CREATE INDEX IF NOT EXISTS idx_combined_score ON matches(combined_similarity_score);
CREATE INDEX IF NOT EXISTS idx_matches_job_rank ON matches(job_id, rank);

-- Document embeddings indexes
CREATE INDEX IF NOT EXISTS idx_doc_id_embeddings ON document_embeddings(doc_id);
CREATE INDEX IF NOT EXISTS idx_milvus_id ON document_embeddings(milvus_id);

-- Job status indexes
CREATE INDEX IF NOT EXISTS idx_status ON job_status(status);
CREATE INDEX IF NOT EXISTS idx_created_at_job_status ON job_status(created_at);

-- Analysis results indexes
CREATE INDEX IF NOT EXISTS idx_job_id_results ON analysis_results(job_id);

-- Batch jobs indexes
CREATE INDEX IF NOT EXISTS idx_status_batch ON batch_jobs(status);
CREATE INDEX IF NOT EXISTS idx_created_at_batch ON batch_jobs(created_at);

-- Metrics indexes
CREATE INDEX IF NOT EXISTS idx_metric_name ON metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_recorded_at ON metrics(recorded_at);

-- Logs indexes
CREATE INDEX IF NOT EXISTS idx_job_id_logs ON logs(job_id);
CREATE INDEX IF NOT EXISTS idx_level ON logs(level);
CREATE INDEX IF NOT EXISTS idx_created_at_logs ON logs(created_at);

-- Document corpus table (for corpus feature)
CREATE TABLE IF NOT EXISTS document_corpus (
    corpus_id SERIAL PRIMARY KEY,
    doc_id INTEGER UNIQUE REFERENCES documents(doc_id) ON DELETE CASCADE,
    tags TEXT[],
    category VARCHAR(100) DEFAULT 'general',
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_doc_id_corpus ON document_corpus(doc_id);
CREATE INDEX IF NOT EXISTS idx_category_corpus ON document_corpus(category);
CREATE INDEX IF NOT EXISTS idx_added_at_corpus ON document_corpus(added_at);

-- Create materialized view for statistics
CREATE VIEW document_stats AS
SELECT
    COUNT(*) as total_documents,
    AVG(file_size) as avg_file_size,
    MAX(file_size) as max_file_size,
    MIN(file_size) as min_file_size
FROM documents;

CREATE VIEW analysis_stats AS
SELECT
    COUNT(*) as total_analyses,
    AVG(ai_score) as avg_ai_score,
    AVG(num_matches) as avg_matches_per_analysis,
    MAX(created_at) as last_analysis_time
FROM analyses;

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO graphplag_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO graphplag_user;

-- Add comments for documentation
COMMENT ON TABLE documents IS 'Stores document metadata and file information';
COMMENT ON TABLE analyses IS 'Stores plagiarism analysis results for documents';
COMMENT ON TABLE matches IS 'Stores individual plagiarism matches found during analysis';
COMMENT ON TABLE document_embeddings IS 'Links documents to their vector embeddings in Milvus';
COMMENT ON TABLE job_status IS 'Tracks processing status of analysis jobs';
COMMENT ON TABLE batch_jobs IS 'Tracks Spark batch processing jobs';
COMMENT ON TABLE metrics IS 'Stores system performance metrics';

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers to auto-update updated_at
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_analyses_updated_at BEFORE UPDATE ON analyses
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_job_status_updated_at BEFORE UPDATE ON job_status
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create initial sample data (optional)
-- This would be populated by the application during normal operation
