USING PERIODIC COMMIT 5000
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row
WHERE row.type IS NOT NULL AND row.type <> ''
MERGE (n:Entity {id: row[':ID']})
SET n.name = row.name,
    n.type = row.type,
    n.title = CASE WHEN row.title <> '' THEN row.title ELSE null END,
    n.year = CASE WHEN row['year:long'] <> '' THEN toInteger(row['year:long']) ELSE null END,
    n.citation_count = CASE WHEN row['citation_count:long'] <> '' THEN toInteger(row['citation_count:long']) ELSE null END,
    n.reference_count = CASE WHEN row['reference_count:long'] <> '' THEN toInteger(row['reference_count:long']) ELSE null END,
    n.repo_stargazers_count = CASE WHEN row['repo_stargazers_count:long'] <> '' THEN toInteger(row['repo_stargazers_count:long']) ELSE null END,
    n.repo_forks_count = CASE WHEN row['repo_forks_count:long'] <> '' THEN toInteger(row['repo_forks_count:long']) ELSE null END,
    n.repo_open_issues_count = CASE WHEN row['repo_open_issues_count:long'] <> '' THEN toInteger(row['repo_open_issues_count:long']) ELSE null END,
    n.repo_updated_at = CASE WHEN row.repo_updated_at <> '' THEN row.repo_updated_at ELSE null END,
    n.repo_created_at = CASE WHEN row.repo_created_at <> '' THEN row.repo_created_at ELSE null END;

USING PERIODIC COMMIT 5000
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
WITH row
WHERE row.type IS NOT NULL AND row.type <> ''
MATCH (n:Entity {id: row[':ID']})
FOREACH (_ IN CASE WHEN row.type = 'Paper' THEN [1] ELSE [] END | SET n:Paper)
FOREACH (_ IN CASE WHEN row.type = 'Method' THEN [1] ELSE [] END | SET n:Method)
FOREACH (_ IN CASE WHEN row.type = 'Repo' THEN [1] ELSE [] END | SET n:Repo)
FOREACH (_ IN CASE WHEN row.type = 'Dataset' THEN [1] ELSE [] END | SET n:Dataset);

USING PERIODIC COMMIT 5000
LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row
WITH row
MATCH (h:Entity {id: row[':START_ID']})
MATCH (t:Entity {id: row[':END_ID']})
WITH h, t, row
FOREACH (_ IN CASE WHEN row[':TYPE'] = 'paper_proposes_method' THEN [1] ELSE [] END |
  MERGE (h)-[r:paper_proposes_method]->(t)
  SET r.doc_id = CASE WHEN row.doc_id <> '' THEN row.doc_id ELSE null END,
      r.source = CASE WHEN row.source <> '' THEN row.source ELSE null END,
      r.confidence = CASE WHEN row['confidence:float'] <> '' THEN toFloat(row['confidence:float']) ELSE null END
)
FOREACH (_ IN CASE WHEN row[':TYPE'] = 'repo_implements_method' THEN [1] ELSE [] END |
  MERGE (h)-[r:repo_implements_method]->(t)
  SET r.doc_id = CASE WHEN row.doc_id <> '' THEN row.doc_id ELSE null END,
      r.source = CASE WHEN row.source <> '' THEN row.source ELSE null END,
      r.confidence = CASE WHEN row['confidence:float'] <> '' THEN toFloat(row['confidence:float']) ELSE null END
)
FOREACH (_ IN CASE WHEN row[':TYPE'] = 'method_uses_dataset' THEN [1] ELSE [] END |
  MERGE (h)-[r:method_uses_dataset]->(t)
  SET r.doc_id = CASE WHEN row.doc_id <> '' THEN row.doc_id ELSE null END,
      r.source = CASE WHEN row.source <> '' THEN row.source ELSE null END,
      r.confidence = CASE WHEN row['confidence:float'] <> '' THEN toFloat(row['confidence:float']) ELSE null END
)
FOREACH (_ IN CASE WHEN row[':TYPE'] = 'paper_cites_paper' THEN [1] ELSE [] END |
  MERGE (h)-[r:paper_cites_paper]->(t)
  SET r.doc_id = CASE WHEN row.doc_id <> '' THEN row.doc_id ELSE null END,
      r.source = CASE WHEN row.source <> '' THEN row.source ELSE null END,
      r.confidence = CASE WHEN row['confidence:float'] <> '' THEN toFloat(row['confidence:float']) ELSE null END
);

