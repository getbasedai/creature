use crate::models::types::{CellContext, Coordinates, Plan, PlanStatus, ColonyStatistics, Thought, DimensionalPosition};
use crate::models::state::LeniaWorldState;
use crate::utils::logging::*;
use std::error::Error;
use std::path::Path;
use std::collections::VecDeque;
use crate::models::plan_analysis::{PlanAnalysis, save_plan_to_file};
use crate::models::constants::{MAX_THOUGHTS_FOR_PLAN, NEIGHBOR_DISTANCE_THRESHOLD, BATCH_SIZE};
use crate::api::openrouter::OpenRouterClient;
use crate::systems::cell::Cell;
use crate::systems::{LeniaWorld, LeniaParams};
use ndarray::Array3;
use std::collections::HashMap;
use chrono::Utc;
use uuid::Uuid;
use rand::{Rng, seq::SliceRandom};

pub struct Colony {
    pub cells: HashMap<Uuid, Cell>,
    pub mission: String,
    pub api_client: OpenRouterClient,
    pub cell_positions: HashMap<Uuid, Coordinates>,
}

impl Colony {

    pub async fn process_cell_sub_batch(&mut self, cell_ids: &[Uuid]) -> Result<(), Box<dyn Error>> {
        let thoughts: Vec<_> = cell_ids.iter()
            .filter_map(|id| self.cells.get(&id))
            .flat_map(|cell| cell.thoughts.iter())
            .collect();

        let mut rng = rand::thread_rng();
        let sampled_thoughts: Vec<String> = thoughts
            .choose_multiple(&mut rng, 3.min(thoughts.len()))
            .map(|thought| thought.content.clone())
            .collect();
        
        log_info(&format!("Processing sub-batch of {} cells with {} sampled thoughts", 
            cell_ids.len(), sampled_thoughts.len()));
        
        let _real_time_context = self.api_client.gather_real_time_context(Some(sampled_thoughts)).await?;
        
        let mut success_count = 0;
        let mut error_count = 0;
        let mut timeout_count = 0;

        // Process each cell directly in self.cells
        for &cell_id in cell_ids {
            if let Some(cell) = self.cells.get_mut(&cell_id) {
                let mut retries = 3;
                let mut delay = std::time::Duration::from_secs(1);
                
                while retries > 0 {
                    match tokio::time::timeout(
                        std::time::Duration::from_secs(180),
                        cell.generate_thought(&self.api_client, &self.mission)
                    ).await {
                        Ok(Ok(_)) => {
                            success_count += 1;
                            log_success(&format!("Generated thought for cell {}", cell_id));
                            break;
                        }
                        Ok(Err(e)) => {
                            if e.to_string().contains("unexpected EOF during chunk size line") {
                                log_warning(&format!("EOF error for cell {}, retrying... ({} attempts left)", cell_id, retries - 1));
                                retries -= 1;
                                if retries > 0 {
                                    tokio::time::sleep(delay).await;
                                    delay *= 2; // Exponential backoff
                                    continue;
                                }
                            }
                            error_count += 1;
                            log_error(&format!("Error generating thought for cell {}: {}", cell_id, e));
                            break;
                        }
                        Err(_) => {
                            timeout_count += 1;
                            log_error(&format!("Timeout generating thought for cell {}", cell_id));
                            break;
                        }
                    }
                }
                
                if retries == 0 {
                    error_count += 1;
                    log_error(&format!("Failed to generate thought for cell {} after all retries", cell_id));
                }
            } else {
                log_error(&format!("Cell {} not found", cell_id));
            }
        }

        log_info(&format!("Batch processing complete: {} successes, {} errors, {} timeouts",
            success_count, error_count, timeout_count));

        Ok(())
    }


    pub fn new(mission: &str, api_client: OpenRouterClient) -> Self {
        Self {
            cells: HashMap::new(),
            mission: mission.to_string(),
            api_client,
            cell_positions: HashMap::new(),
        }
    }

    fn analyze_dimensional_balance(&self, cell_ids: &[Uuid]) -> (DimensionalPosition, f64) {
        let mut combined = DimensionalPosition {
            emergence: 0.0,
            coherence: 0.0,
            resilience: 0.0,
            intelligence: 0.0,
            efficiency: 0.0,
            integration: 0.0,
        };
        
        let mut total_cells = 0;
        
        // Calculate average position
        for &id in cell_ids {
            if let Some(cell) = self.cells.get(&id) {
                combined.emergence += cell.dimensional_position.emergence;
                combined.coherence += cell.dimensional_position.coherence;
                combined.resilience += cell.dimensional_position.resilience;
                combined.intelligence += cell.dimensional_position.intelligence;
                combined.efficiency += cell.dimensional_position.efficiency;
                combined.integration += cell.dimensional_position.integration;
                total_cells += 1;
            }
        }
        
        if total_cells > 0 {
            combined.emergence /= total_cells as f64;
            combined.coherence /= total_cells as f64;
            combined.resilience /= total_cells as f64;
            combined.intelligence /= total_cells as f64;
            combined.efficiency /= total_cells as f64;
            combined.integration /= total_cells as f64;
        }
        
        // Calculate imbalance score (0 = perfect balance, higher = more imbalanced)
        let imbalance = (combined.emergence.abs() + 
                        combined.coherence.abs() + 
                        combined.resilience.abs() +
                        combined.intelligence.abs() +
                        combined.efficiency.abs() +
                        combined.integration.abs()) / 6.0;
                        
        (combined, imbalance)
    }

    pub async fn process_cell_batch(&mut self, cell_ids: &[Uuid]) -> Result<(), Box<dyn std::error::Error>> {
        use crate::utils::logging::*;
        use tokio::time::timeout;
        use std::time::Duration;
        
        
        log_timestamp(&format!("Starting batch processing of {} cells", cell_ids.len()));
        
        let batch_id = Uuid::new_v4();
        log_metric("Batch ID", batch_id);
        log_metric("Processing Mode", "Batch Evolution");
        log_metric("System Status", "Active");
        log_memory_usage("Memory Usage", std::mem::size_of::<Cell>() * self.cells.len());
        
        // Analyze dimensional balance
        let (current_position, imbalance) = self.analyze_dimensional_balance(cell_ids);
        println!("║ Batch Details:");
        println!("║   Size: {} cells", cell_ids.len());
        println!("║   Dimensional Imbalance: {:.2}", imbalance);
        println!("║   Processing Start: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S.%3f"));
        println!("║   Batch Memory: {} bytes", std::mem::size_of::<Cell>() * cell_ids.len());
        println!("║ Dimensional Analysis:");
        println!("║   Current Position (normalized):");
        println!("║   ├─ Emergence: {:.2} [{:>3}%]", 
            current_position.emergence,
            ((current_position.emergence + 100.0) / 2.0).round()
        );
        println!("║   Coherence: {:.2}", current_position.coherence);
        println!("║   Resilience: {:.2}", current_position.resilience);
        println!("║   Intelligence: {:.2}", current_position.intelligence);
        println!("║   Efficiency: {:.2}", current_position.efficiency);
        println!("║   Integration: {:.2}", current_position.integration);
        
        // Process cells in smaller sub-batches
                // Process cells in smaller sub-batches
        let sub_batch_size = 5;
        let _results: Vec<Result<(), Box<dyn std::error::Error>>> = Vec::new();
        
        for chunk in cell_ids.chunks(sub_batch_size) {
            let mut retries = 3;
            let mut delay = Duration::from_secs(1);
            
            while retries > 0 {
                match timeout(
                    Duration::from_secs(300),
                    self.process_cell_sub_batch(chunk)
                ).await {
                    Ok(Ok(_)) => {
                        break;
                    }
                    Ok(Err(e)) => {
                        log_warning(&format!("Error in sub-batch: {}. Retries left: {}", e, retries - 1));
                        retries -= 1;
                        if retries > 0 {
                            tokio::time::sleep(delay).await;
                            delay *= 2; // Exponential backoff
                        }
                    }
                    Err(_) => {
                        log_warning(&format!("Timeout in sub-batch. Retries left: {}", retries - 1));
                        retries -= 1;
                        if retries > 0 {
                            tokio::time::sleep(delay).await;
                            delay *= 2;
                        }
                    }
                }
            }
            
            if retries == 0 {
                log_error("Failed to process sub-batch after all retries");
                continue;
            }
        }
        println!("║ Context Gathering Phase:");
        println!("║   Start Time: {}", chrono::Local::now().format("%H:%M:%S.%3f"));
        println!("║   Status: Active");
        println!("║   Mode: Real-time Analysis");
        let real_time_context = self.api_client.gather_real_time_context(None).await?;
        println!("║ Context Analysis Complete:");
        println!("║   Time: {}", chrono::Local::now().format("%H:%M:%S.%3f"));
        println!("║   Market Trends: {} identified", real_time_context.market_trends.len());
        println!("║   Tech Developments: {} analyzed", real_time_context.technological_developments.len());
        println!("║   Current Events: {} processed", real_time_context.current_events.len());
        println!("║   User Interactions: {} tracked", real_time_context.user_interactions.len());
        println!("║   Context Memory: {} bytes", 
            serde_json::to_string(&real_time_context)
                .map(|s| s.len())
                .unwrap_or(0)
        );
        
        // Sort cells by dimensional scores and prepare contexts
        let mut ranked_cells: Vec<(&Uuid, &Cell)> = cell_ids.iter()
            .filter_map(|id| self.cells.get(id).map(|cell| (id, cell)))
            .collect();

        // Rank cells by their dimensional balance
        ranked_cells.sort_by(|(_id1, cell1), (_id2, cell2)| {
            let score1 = (cell1.dimensional_position.emergence.abs() +
                         cell1.dimensional_position.coherence.abs() +
                         cell1.dimensional_position.resilience.abs() +
                         cell1.dimensional_position.intelligence.abs() +
                         cell1.dimensional_position.efficiency.abs() +
                         cell1.dimensional_position.integration.abs()) / 6.0;
            
            let score2 = (cell2.dimensional_position.emergence.abs() +
                         cell2.dimensional_position.coherence.abs() +
                         cell2.dimensional_position.resilience.abs() +
                         cell2.dimensional_position.intelligence.abs() +
                         cell2.dimensional_position.efficiency.abs() +
                         cell2.dimensional_position.integration.abs()) / 6.0;
            
            score1.partial_cmp(&score2).unwrap()
        });

        // Prepare batch of cell contexts, maintaining ranked order
        let cell_contexts: Vec<(Uuid, CellContext)> = ranked_cells.iter()
            .map(|(id, cell)| (**id, CellContext {
                current_focus: cell.get_current_focus(),
                active_research_topics: cell.get_active_research(),
                recent_discoveries: cell.get_recent_discoveries(),
                collaboration_history: cell.get_collaboration_history(),
                performance_metrics: cell.get_performance_metrics(),
                evolution_stage: cell.get_evolution_stage(),
                energy_level: cell.energy,
                dimensional_position: cell.dimensional_position.clone(),
                dopamine: cell.dopamine,
            }))
            .collect();

        // Print dimensional rankings
        println!("║ Dimensional Rankings:");
        for (i, (_id, cell)) in ranked_cells.iter().enumerate() {
            println!("║   Cell {}: [E:{:.2} C:{:.2} R:{:.2} I:{:.2} Ef:{:.2} In:{:.2}]",
                i + 1,
                cell.dimensional_position.emergence,
                cell.dimensional_position.coherence,
                cell.dimensional_position.resilience,
                cell.dimensional_position.intelligence,
                cell.dimensional_position.efficiency,
                cell.dimensional_position.integration
            );
        }

        let cell_context_refs: Vec<(Uuid, &CellContext)> = cell_contexts.iter()
            .map(|(id, context)| (*id, context))
            .collect();

        println!("║ Gathering real-time context for {} cells...", cell_contexts.len());
        
        // Get batch of thoughts from LLM with timeout
        println!("╠════════════════════════════════════════════════════════════╣");
        
        println!("║ [{}] Generating thoughts...", 
            chrono::Local::now().format("%H:%M:%S"));
        let batch_results = match tokio::time::timeout(
            std::time::Duration::from_secs(300), // Reduced timeout
            self.api_client.generate_contextual_thoughts_batch(&cell_context_refs, &real_time_context, &self.mission)
        ).await {
            Ok(result) => match result {
                Ok(batch) => batch,
                Err(e) => {
                    eprintln!("Error generating thoughts: {}", e);
                    HashMap::new() // Return empty results on error
                }
            },
            Err(_) => {
                eprintln!("Thought generation timed out after 300 seconds");
                HashMap::new()
            }
        };
        
        println!("║   Thoughts Generated: {}", batch_results.len());
        println!("║   Success Rate: {:.1}%", 
            (batch_results.len() as f64 / cell_contexts.len() as f64) * 100.0);

        // Update cells with their new thoughts, adjusting dimensional positions 
        for (cell_id, (thought_content, relevance_score, real_time_factors)) in batch_results {
            // First get a clone of the cell
            if let Some(cell) = self.cells.get(&cell_id).cloned() {
                let mut updated_cell = cell;
                
                // Adjust dimensional position based on imbalance
                let adjustment = 0.1 * (1.0 - imbalance.min(1.0));
                updated_cell.dimensional_position.resilience += adjustment * -updated_cell.dimensional_position.resilience.signum();
                updated_cell.dimensional_position.intelligence += adjustment * -updated_cell.dimensional_position.intelligence.signum();
                updated_cell.dimensional_position.efficiency += adjustment * -updated_cell.dimensional_position.efficiency.signum();
                updated_cell.dimensional_position.integration += adjustment * -updated_cell.dimensional_position.integration.signum();
                
                let thought = Thought {
                    id: Uuid::new_v4(),
                    content: thought_content.clone(),
                    timestamp: Utc::now(),
                    relevance_score,
                    context_tags: updated_cell.generate_context_tags(&CellContext {
                        current_focus: updated_cell.get_current_focus(),
                        active_research_topics: updated_cell.get_active_research(),
                        recent_discoveries: updated_cell.get_recent_discoveries(),
                        collaboration_history: updated_cell.get_collaboration_history(),
                        performance_metrics: updated_cell.get_performance_metrics(),
                        evolution_stage: updated_cell.get_evolution_stage(),
                        energy_level: updated_cell.energy,
                        dimensional_position: updated_cell.dimensional_position.clone(),
                        dopamine: updated_cell.dopamine,
                    }),
                    real_time_factors,
                    confidence_score: updated_cell.calculate_confidence_score(&real_time_context),
                };

                // Parse dimensional positions from thought content
                for line in thought_content.lines() {
                    let line = line.trim();
                    if line.starts_with("DIMENSIONS:") {
                        continue;
                    }
                    if line.starts_with("- EMERGENT_INTELLIGENCE:") {
                        updated_cell.dimensional_position.emergence = line.split(':')
                            .nth(1)
                            .and_then(|s| s.trim().parse().ok())
                            .unwrap_or(0.0);
                    } else if line.starts_with("- RESOURCE_EFFICIENCY:") {
                        updated_cell.dimensional_position.efficiency = line.split(':')
                            .nth(1)
                            .and_then(|s| s.trim().parse().ok())
                            .unwrap_or(0.0);
                    } else if line.starts_with("- NETWORK_COHERENCE:") {
                        updated_cell.dimensional_position.coherence = line.split(':')
                            .nth(1)
                            .and_then(|s| s.trim().parse().ok())
                            .unwrap_or(0.0);
                    } else if line.starts_with("- GOAL_ALIGNMENT:") {
                        updated_cell.dimensional_position.intelligence = line.split(':')
                            .nth(1)
                            .and_then(|s| s.trim().parse().ok())
                            .unwrap_or(0.0);
                    } else if line.starts_with("- TEMPORAL_RESILIENCE:") {
                        updated_cell.dimensional_position.resilience = line.split(':')
                            .nth(1)
                            .and_then(|s| s.trim().parse().ok())
                            .unwrap_or(0.0);
                    } else if line.starts_with("- DIMENSIONAL_INTEGRATION:") {
                        updated_cell.dimensional_position.integration = line.split(':')
                            .nth(1)
                            .and_then(|s| s.trim().parse().ok())
                            .unwrap_or(0.0);
                    } else if line.starts_with("DOPAMINE:") {
                        updated_cell.dopamine = line.split(':')
                            .nth(1)
                            .and_then(|s| s.trim().parse().ok())
                            .unwrap_or(0.5);
                    }
                }
                
                updated_cell.thoughts.push_back(thought);
                if let Err(e) = updated_cell.check_and_compress_memories(&self.api_client).await {
                    eprintln!("Error compressing memories: {}", e);
                }
                
                // Finally insert the updated cell back into the HashMap
                self.cells.insert(cell_id, updated_cell);
            }
        }

        Ok(())
    }

    pub async fn create_plans_batch(&mut self, cell_ids: &[Uuid], cycle_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        use crate::utils::logging::*;
        
        log_timestamp(&format!("Starting plan generation for {} cells", cell_ids.len()));
        
        let batch_id = Uuid::new_v4();
        log_metric("Batch ID", batch_id);
        log_metric("Total Cells", cell_ids.len());
        log_metric("Current Time", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));
        log_memory_usage("Batch Memory", std::mem::size_of::<Cell>() * cell_ids.len());
        
        let mut updates = Vec::new();
        let mut best_plan_score = 0.0;
        let mut best_plan_narrative = String::new();
        let start_time = std::time::Instant::now();
        
        for &cell_id in cell_ids {
            if let Some(cell) = self.cells.get(&cell_id).cloned() {
                // Calculate dimensional differences between cells
                let mut neighbor_scores: Vec<(Uuid, f64)> = cell.neighbors.iter()
                    .filter_map(|&neighbor_id| {
                        self.cells.get(&neighbor_id).map(|neighbor| {
                            let score = calculate_dimensional_complement(
                                &cell.dimensional_position,
                                &neighbor.dimensional_position
                            );
                            (neighbor_id, score)
                        })
                    })
                    .collect();

                // Sort neighbors by complementary score (higher is better)
                neighbor_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                // Collect thoughts prioritizing complementary cells
                let mut combined_thoughts = Vec::new();
                combined_thoughts.extend(cell.thoughts.iter().cloned());

                for (neighbor_id, _score) in neighbor_scores {
                    if let Some(neighbor) = self.cells.get(&neighbor_id) {
                        // Weight thoughts by how well they complement the cell's dimensions
                        let mut neighbor_thoughts: Vec<_> = neighbor.thoughts.iter()
                            .map(|thought| {
                                let dimensional_weight = calculate_dimensional_complement(
                                    &cell.dimensional_position,
                                    &neighbor.dimensional_position
                                );
                                (thought.clone(), thought.relevance_score * dimensional_weight)
                            })
                            .collect();
                        
                        neighbor_thoughts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        combined_thoughts.extend(neighbor_thoughts.into_iter().map(|(t, _)| t));
                    }
                }

                combined_thoughts.truncate(MAX_THOUGHTS_FOR_PLAN);

                println!("║ Creating plan for cell {}...", cell_id);
                                let plan_result = match tokio::time::timeout(
                    std::time::Duration::from_secs(300), // Reduced timeout
                    self.api_client.create_plan(&combined_thoughts)
                ).await {
                    Ok(result) => match result {
                        Ok(plan) => plan,
                        Err(e) => {
                            eprintln!("Error creating plan: {}", e);
                            continue; // Skip this cell on error
                        }
                    },
                    Err(_) => {
                        eprintln!("Plan creation timed out after 300 seconds");
                        continue;
                    }
                };
                
                println!("║ Cell Details:");
                println!("║   ID: {}", cell_id);
                println!("║   Energy Level: {:.2}", cell.energy);
                println!("║   Position: ({:.2}, {:.2}, {:.2})", 
                    cell.position.x, cell.position.y, cell.position.z);
                println!("║   Thoughts: {}", combined_thoughts.len());
                println!("║   Neighbors: {}", cell.neighbors.len());
                println!("║   Dimensional Position:");
                println!("║     Emergence: {:.2}", cell.dimensional_position.emergence);
                println!("║     Coherence: {:.2}", cell.dimensional_position.coherence);
                println!("║     Resilience: {:.2}", cell.dimensional_position.resilience);
                println!("║     Intelligence: {:.2}", cell.dimensional_position.intelligence);
                println!("║     Efficiency: {:.2}", cell.dimensional_position.efficiency);
                println!("║     Integration: {:.2}", cell.dimensional_position.integration);
                println!("║");
                println!("║   Plan Details:");
                println!("║     Nodes: {}", plan_result.nodes.len());
                println!("║     Score: {:.2}", plan_result.score);
                println!("║     Participating Cells: {}", plan_result.participating_cells.len());
                
                // Log the full plan details
                
                let plan_score = combined_thoughts.iter().map(|t| t.relevance_score).sum::<f64>() / combined_thoughts.len() as f64;
                
                let plan = Plan {
                    id: Uuid::new_v4(),
                    thoughts: combined_thoughts.clone(),
                    nodes: plan_result.nodes.clone(),
                    summary: plan_result.summary.clone(),
                    score: plan_score,
                    participating_cells: std::iter::once(cell_id)
                        .chain(cell.neighbors.iter().cloned())
                        .collect(),
                    created_at: Utc::now(),
                    status: PlanStatus::Proposed,
                };

                if plan_score > best_plan_score {
                    best_plan_score = plan_score;
                    best_plan_narrative = plan_result.summary.clone();
                }

                for &participant_id in &plan.participating_cells {
                    if let Some(mut participant) = self.cells.get(&participant_id).cloned() {
                        participant.current_plan = Some(plan.clone());
                        updates.push(participant);
                    }
                }
            }
        }

        let elapsed = start_time.elapsed();
        println!("╠════════════════════════════════════════════════════════════╣");
        println!("║ Batch Summary:");
        println!("║   Total plans created: {}", updates.len());
        println!("║   Best plan score: {:.2}", best_plan_score);
        println!("║   Processing time: {:.2}s", elapsed.as_secs_f64());
        println!("╚════════════════════════════════════════════════════════════╝");

        // Query for relevant news about the best plan
        if !best_plan_narrative.is_empty() {
            println!("\nAnalyzing technical context for best plan (score: {:.2})...", best_plan_score);
            let news_query = format!(
                r#"Analyze this AI system's plan against posts from the last 72 hours (2024 only) by technical accounts (<0.01% X following):

                PLAN:
                {}

                Find posts from last 72 hours about:
                1. Technical developments directly related to plan components
                2. New tools/methods that could enhance plan execution
                3. Similar approaches being tested by research labs
                4. Technical limitations discovered by researchers
                
                Requirements:
                - Posts must be from last 72 hours in 2024
                - Only from accounts with <0.01% X following
                - Include exact timestamps and account names
                - Link to any referenced papers/code
                - Technical content only, no general news"#,
                best_plan_narrative
            );

            match self.api_client.query_llm(&news_query).await {
                Ok(news) => println!("\nRelevant developments:\n{}", news),
                Err(e) => eprintln!("Error querying for relevant news: {}", e),
            }
        }

        // Save all plans to disk
        let plans_path = std::path::Path::new("data/plans");
        for cell in &updates {
            if let Some(plan) = &cell.current_plan {
                save_plan_to_file(plan, plans_path, cycle_id)?;
            }
        }

        // Create and save plan analysis
        let all_plans: Vec<Plan> = updates.iter()
            .filter_map(|cell| cell.current_plan.clone())
            .collect();
            
        let analysis = PlanAnalysis::analyze_plans(&all_plans, cycle_id);
        analysis.save_to_file(plans_path)?;

        // Update cells
        for cell in updates {
            self.cells.insert(cell.id, cell);
        }

        Ok(())
    }
    pub fn add_cell(&mut self, position: Coordinates) -> Uuid {
        let mut cell = Cell::new(position.clone());
        let id = cell.id;
        self.cells.insert(id, cell);
        self.cell_positions.insert(id, position);
        self.update_neighbors(id);
        id
    }

    fn update_neighbors(&mut self, cell_id: Uuid) {
        let cell_pos = self.cells.get(&cell_id).unwrap().position.clone();
        let mut neighbors = Vec::new();

        for (other_id, other_cell) in self.cells.iter() {
            if *other_id != cell_id {
                let distance = calculate_distance(&cell_pos, &other_cell.position);
                if distance < NEIGHBOR_DISTANCE_THRESHOLD {
                    neighbors.push(*other_id);
                }
            }
        }

        if let Some(cell) = self.cells.get_mut(&cell_id) {
            cell.neighbors = neighbors;
        }
    }

    pub async fn handle_cell_reproduction(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n[{}] Starting cell reproduction cycle", 
            chrono::Local::now().format("%H:%M:%S"));
        let mut new_cells = Vec::new();
        let mut rng = rand::thread_rng();
        
        for cell in self.cells.values() {
            if cell.energy > 90.0 && rng.gen::<f64>() < 0.1 {
                let mut new_position = Coordinates {
                    x: cell.position.x + (rng.gen::<f64>() - 0.5),
                    y: cell.position.y + (rng.gen::<f64>() - 0.5),
                    z: cell.position.z + (rng.gen::<f64>() - 0.5),
                    heat: cell.position.heat * (0.9 + rng.gen::<f64>() * 0.2), // Inherit with slight variation
                    emergence_score: cell.position.emergence_score + (rng.gen::<f64>() * 10.0 - 5.0),
                    coherence_score: cell.position.coherence_score + (rng.gen::<f64>() * 10.0 - 5.0),
                    resilience_score: cell.position.resilience_score + (rng.gen::<f64>() * 10.0 - 5.0),
                    intelligence_score: cell.position.intelligence_score + (rng.gen::<f64>() * 10.0 - 5.0),
                    efficiency_score: cell.position.efficiency_score + (rng.gen::<f64>() * 10.0 - 5.0),
                    integration_score: cell.position.integration_score + (rng.gen::<f64>() * 10.0 - 5.0),
                };
                
                // Clamp all scores to valid ranges
                new_position.heat = new_position.heat.clamp(0.0, 1.0);
                new_position.emergence_score = new_position.emergence_score.clamp(-100.0, 100.0);
                new_position.coherence_score = new_position.coherence_score.clamp(-100.0, 100.0);
                new_position.resilience_score = new_position.resilience_score.clamp(-100.0, 100.0);
                new_position.intelligence_score = new_position.intelligence_score.clamp(-100.0, 100.0);
                new_position.efficiency_score = new_position.efficiency_score.clamp(-100.0, 100.0);
                new_position.integration_score = new_position.integration_score.clamp(-100.0, 100.0);
                new_cells.push(new_position);
            }
        }

        println!("[{}] Creating {} new cells through reproduction", 
            chrono::Local::now().format("%H:%M:%S"),
            new_cells.len());
        
        for position in new_cells {
            self.add_cell(position);
        }
        
        println!("[{}] Cell reproduction cycle completed", 
            chrono::Local::now().format("%H:%M:%S"));

        Ok(())
    }

    pub async fn update_mission_progress(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Implementation for mission progress update
        Ok(())
    }

    pub async fn compress_colony_memories(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        for cell in self.cells.values_mut() {
            cell.check_and_compress_memories(&self.api_client).await?;
        }
        Ok(())
    }

    pub fn print_cycle_statistics(&self, cycle: i32) {
        println!("\n");
        println!("                      Cycle {} Statistics                     ", cycle);
        println!("--------------------------------------------------------------");
        
        let mut total_energy = 0.0;
        let mut total_thoughts = 0;
        let mut total_compressed_memories = 0;
        let mut cells_with_plans = 0;
        
        for cell in self.cells.values() {
            total_energy += cell.energy;
            total_thoughts += cell.thoughts.len();
            total_compressed_memories += cell.compressed_memories.len();
            if cell.current_plan.is_some() {
                cells_with_plans += 1;
            }
        }

        println!("║ ┌──────────────────────────┬───────────────────────────┐ ║");
        println!("║ │         Metric           │          Value            │ ║");
        println!("║ ├──────────────────────────┼───────────────────────────┤ ║");
        println!("║ │ Active Cells             │ {:<21} │ ║", self.cells.len());
        println!("║ │ Average Energy           │ {:<21.2} │ ║", total_energy / self.cells.len() as f64);
                println!("║ │ Total Thoughts           │ {:<21} │ ║", total_thoughts);
        println!("║ │ Cells with Active Plans  │ {:<21} │ ║", cells_with_plans);
        println!("║ │ Compressed Memory Blocks │ {:<21} │ ║", total_compressed_memories);
        println!("║ └──────────────────────────┴───────────────────────────┘ ║");
        println!("╚════════════════════════════════════════════════════════════╝");
    }

    pub fn print_memory_statistics(&self) {
        let mut total_thought_size = 0;
        let mut total_compressed_size = 0;
        
        for cell in self.cells.values() {
            total_thought_size += cell.thoughts.iter()
                .map(|t| t.content.len())
                .sum::<usize>();
            
            total_compressed_size += cell.compressed_memories.iter()
                .map(|m| m.len())
                .sum::<usize>();
        }

        let compression_ratio = if total_compressed_size > 0 {
            total_thought_size as f64 / total_compressed_size as f64
        } else {
            0.0
        };

        println!("\n");
        println!("                    Memory Statistics                         ");
        println!("------------------------------------------------------------");
        println!("║ ┌──────────────────────────┬───────────────────────────┐ ║");
        println!("║ │         Metric           │          Value            │ ║");
        println!("║ ├──────────────────────────┼───────────────────────────┤ ║");
        println!("║ │ Total Thought Memory     │ {:<21} │ ║", format!("{} bytes", total_thought_size));
        println!("║ │ Total Compressed Memory  │ {:<21} │ ║", format!("{} bytes", total_compressed_size));
        println!("║ │ Compression Ratio        │ {:<21.2}x │ ║", compression_ratio);
        println!("║ └──────────────────────────┴───────────────────────────┘ ║");
        println!("╚════════════════════════════════════════════════════════════╝");
    }

    pub async fn evolve_cells(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        use crate::utils::logging::*;
        use tokio::sync::Semaphore;
        use std::sync::Arc;
        use futures::future::join_all;
        
        log_timestamp("Starting cell evolution cycle");
        
        let evolution_id = Uuid::new_v4();
        log_metric("Evolution ID", evolution_id);
        log_metric("Active Cells", self.cells.len());
        log_metric("Average Energy", format!("{:.2}", self.get_average_energy()));
        log_metric("Total Thoughts", self.get_total_thoughts());
        log_metric("Mutation Rate", format!("{:.2}%", self.get_mutation_rate() * 100.0));
        println!("║ Evolution ID: {}", evolution_id);
        
        // Add detailed logging for Lenia simulation
        println!("[{}] Preparing to advance Lenia simulation...", 
            chrono::Local::now().format("%H:%M:%S"));
        
        // Log colony state before evolution
        println!("║ Colony State Before Evolution:");
        println!("║   Total Cells: {}", self.cells.len());
        println!("║   Active Cells: {}", 
            self.cells.values().filter(|c| c.energy > 50.0).count());
        println!("║   Total Energy: {:.2}", 
            self.cells.values().map(|c| c.energy).sum::<f64>());
        
        // Simple position-based evolution
        for (id, pos) in self.cell_positions.iter_mut() {
            if let Some(cell) = self.cells.get_mut(id) {
                // Update cell energy based on position
                let position_factor = (pos.x.abs() + pos.y.abs() + pos.z.abs()) / 30.0;
                cell.energy = (cell.energy + position_factor).clamp(0.0, 100.0);
                
                // Update stability based on neighbor count
                let neighbor_count = cell.neighbors.len() as f64;
                cell.stability = (cell.stability * 0.9 + (neighbor_count / 10.0).min(1.0) * 0.1)
                    .clamp(0.0, 1.0);
            }
        }
                let cell_ids: Vec<Uuid> = self.cells.keys().copied().collect();
        let total_cells = cell_ids.len();
        
        println!("\n");
        println!("                      Evolution Batch                          ");
        println!("--------------------------------------------------------------");

        // Create a semaphore to limit concurrent tasks
        let semaphore = Arc::new(Semaphore::new(4)); // Limit to 4 concurrent batches
        let mut tasks = Vec::new();

        for batch_idx in (0..cell_ids.len()).step_by(BATCH_SIZE) {
            let batch_end = (batch_idx + BATCH_SIZE).min(cell_ids.len());
            let batch: Vec<Uuid> = cell_ids[batch_idx..batch_end].to_vec();
            
            // Clone necessary data for the task
            let sem_clone = semaphore.clone();
            let batch_cells: HashMap<Uuid, Cell> = batch.iter()
                .filter_map(|id| self.cells.get(id).map(|cell| (*id, cell.clone())))
                .collect();
            
            println!("\n[Batch {}/{}]", (batch_idx / BATCH_SIZE) + 1, (cell_ids.len() + BATCH_SIZE - 1) / BATCH_SIZE);
            
            // Spawn a new task for this batch
            let task = tokio::spawn(async move {
                // Acquire semaphore permit
                let _permit = sem_clone.acquire().await.unwrap();
                
                let mut updated_cells = HashMap::new();
                
                for (cell_id, mut cell) in batch_cells {
                    // Get Lenia state at cell's position
                    let lenia_state = cell.lenia_state; // Use cached state
                    
                    // Adjust cell energy based on Lenia state
                    let lenia_influence = lenia_state * cell.lenia_influence;
                    cell.energy = (cell.energy + lenia_influence).clamp(0.0, 100.0);
                    
                    // Base energy regeneration
                    if cell.energy < 50.0 {
                        cell.energy += 10.0;
                    }
                    
                    // Update cell stability based on Lenia state
                    cell.stability = (cell.stability * 0.9 + lenia_state * 0.1).clamp(0.0, 1.0);
                    
                    updated_cells.insert(cell_id, cell);
                }
                
                updated_cells
            });
            
            tasks.push(task);
        }

        // Wait for all tasks to complete and collect results
        let results = join_all(tasks).await;
        
        // Process results and update cells
        for result in results {
            match result {
                Ok(batch_updates) => {
                    for (id, cell) in batch_updates {
                        self.cells.insert(id, cell);
                    }
                }
                Err(e) => {
                    eprintln!("Error in evolution batch: {}", e);
                }
            }
        }

        println!("\n[{}] Evolution cycle completed for {} cells", 
            chrono::Local::now().format("%H:%M:%S"),
            total_cells);
            
        // Perform dimensional position audit
        self.audit_dimensional_positions().await?;
        
        Ok(())
    }

    pub fn save_state(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.save_state_to_file("eca_state.json")?;
        Ok(())
    }

    pub fn save_state_to_file(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        use crate::models::state::{ColonyState, CellState, EnergyGridState};
        
        // Calculate grid size based on cell positions
        let max_coord = self.cells.values()
            .flat_map(|cell| vec![
                (cell.position.x * 2.0) as usize,
                (cell.position.y * 2.0) as usize,
                (cell.position.z * 2.0) as usize
            ])
            .max()
            .unwrap_or(0) + 1;
        
        let grid_size = max_coord + 1;
        let total_size = grid_size * grid_size * grid_size;
        let mut grid = vec![0.0; total_size];
        let mut cell_positions = HashMap::new();
        
        // Map cells to grid positions and store energy values
        for (id, cell) in &self.cells {
            let x = (cell.position.x * 2.0) as usize;
            let y = (cell.position.y * 2.0) as usize;
            let z = (cell.position.z * 2.0) as usize;
            
            let idx = z * grid_size * grid_size + y * grid_size + x;
            if idx < total_size {
                grid[idx] = cell.energy;
                cell_positions.insert(*id, (x, y, z));
            }
        }
        
        let cell_states: HashMap<Uuid, CellState> = self.cells.iter()
            .map(|(id, cell)| {
                (*id, CellState {
                    id: cell.id,
                    energy: cell.energy,
                    thoughts: cell.thoughts.iter().cloned().collect(),
                    current_plan: cell.current_plan.clone(),
                    dimensional_position: cell.dimensional_position.clone(),
                    dopamine: cell.dopamine,
                    stability: cell.stability,
                    phase: cell.phase,
                    context_alignment_score: cell.context_alignment_score,
                    mission_alignment_score: cell.mission_alignment_score,
                    lenia_state: cell.lenia_state,
                    lenia_influence: cell.lenia_influence,
                    x: cell.position.x,
                    y: cell.position.y,
                    z: cell.position.z,
                })
            })
            .collect();

        let state = ColonyState {
            timestamp: Utc::now(),
            cells: cell_states,
            total_cycles: 0,
            mission: self.mission.clone(),
            lenia_world: None,
            energy_grid: EnergyGridState {
                size: grid_size,
                grid,
                cell_positions,
            },
        };

        state.save_to_file(Path::new(filename))?;
        Ok(())
    }

    pub fn load_state_from_file(&mut self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        use crate::models::state::ColonyState;
        
        let state = ColonyState::load_from_file(Path::new(filename))?;
        self.mission = state.mission;
        
        // Clear existing cells and load from state
        self.cells.clear();
        for (id, cell_state) in state.cells {
            if let Some(cell) = self.cells.get_mut(&id) {
                cell.energy = cell_state.energy;
                cell.thoughts = VecDeque::from(cell_state.thoughts);
                cell.current_plan = cell_state.current_plan;
                cell.dimensional_position = cell_state.dimensional_position;
                cell.dopamine = cell_state.dopamine;
                cell.stability = cell_state.stability;
                cell.phase = cell_state.phase;
                cell.context_alignment_score = cell_state.context_alignment_score;
                cell.mission_alignment_score = cell_state.mission_alignment_score;
            }
        }
        
        Ok(())
    }

    pub fn print_statistics(&self) {
        
        
        
        let stats = self.get_statistics();
        println!("║ ┌──────────────────────────┬───────────────────────────┐ ║");
        println!("║ │         Metric           │          Value            │ ║");
        println!("║ ├──────────────────────────┼───────────────────────────┤ ║");
        println!("║ │ Total Cells              │ {:<21} │ ║", stats.total_cells);
        println!("║ │ Total Thoughts           │ {:<21} │ ║", stats.total_thoughts);
        println!("║ │ Total Plans              │ {:<21} │ ║", stats.total_plans);
        println!("║ │ Successful Plans         │ {:<21} │ ║", stats.successful_plans);
        println!("║ │ Failed Plans             │ {:<21} │ ║", stats.failed_plans);
        println!("║ │ Average Cell Energy      │ {:<21.2} │ ║", stats.average_cell_energy);
        println!("║ │ Highest Evolution Stage  │ {:<21} │ ║", stats.highest_evolution_stage);
        println!("║ │ Total Cycles             │ {:<21} │ ║", stats.total_cycles);
        println!("║ └──────────────────────────┴───────────────────────────┘ ║");
        println!("╚════════════════════════════════════════════════════════════╝");

        // Print detailed memory distribution
        let mut total_memory = 0;
        let mut max_memory_cell = None;
        let mut min_memory_cell = None;
        let base_cell_size = std::mem::size_of::<Cell>();

        for (id, cell) in &self.cells {
            let cell_memory = base_cell_size + 
                cell.thoughts.iter().map(|t| t.content.len()).sum::<usize>() +
                cell.compressed_memories.iter().map(|m| m.len()).sum::<usize>();
            total_memory += cell_memory;

            match (max_memory_cell, min_memory_cell) {
                (None, None) => {
                    max_memory_cell = Some((id, cell_memory));
                    min_memory_cell = Some((id, cell_memory));
                }
                (Some((_, max_mem)), Some((_, min_mem))) => {
                    if cell_memory > max_mem {
                        max_memory_cell = Some((id, cell_memory));
                    }
                    if cell_memory < min_mem {
                        min_memory_cell = Some((id, cell_memory));
                    }
                }
                _ => unreachable!()
            }
        }

        println!("\n║ Memory Distribution:");
        println!("║   Total Memory Usage: {} bytes", total_memory);
        println!("║   Average Memory per Cell: {} bytes", 
            if !self.cells.is_empty() { total_memory / self.cells.len() } else { 0 });
                if let Some((id, mem)) = max_memory_cell {
            println!("║   Highest Memory Cell: {} ({} bytes)", id, mem);
        }
        if let Some((id, mem)) = min_memory_cell {
            println!("║   Lowest Memory Cell: {} ({} bytes)", id, mem);
        }

        // Print dimensional statistics
        println!("\n║ Dimensional Statistics:");
        let mut total_pos = DimensionalPosition {
            emergence: 0.0,
            coherence: 0.0,
            resilience: 0.0,
            intelligence: 0.0,
            efficiency: 0.0,
            integration: 0.0,
        };

        for cell in self.cells.values() {
            total_pos.emergence += cell.dimensional_position.emergence;
            total_pos.coherence += cell.dimensional_position.coherence;
            total_pos.resilience += cell.dimensional_position.resilience;
            total_pos.intelligence += cell.dimensional_position.intelligence;
            total_pos.efficiency += cell.dimensional_position.efficiency;
            total_pos.integration += cell.dimensional_position.integration;
        }

        if !self.cells.is_empty() {
            let cell_count = self.cells.len() as f64;
            println!("║   Average Emergence: {:.2}", total_pos.emergence / cell_count);
            println!("║   Average Coherence: {:.2}", total_pos.coherence / cell_count);
            println!("║   Average Resilience: {:.2}", total_pos.resilience / cell_count);
            println!("║   Average Intelligence: {:.2}", total_pos.intelligence / cell_count);
            println!("║   Average Efficiency: {:.2}", total_pos.efficiency / cell_count);
            println!("║   Average Integration: {:.2}", total_pos.integration / cell_count);
        }
    }

    pub async fn audit_dimensional_positions(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n[{}] Starting dimensional position audit...", 
            chrono::Local::now().format("%H:%M:%S"));
            
        for (cell_id, cell) in self.cells.iter_mut() {
            if let Some(plan) = &cell.current_plan {
                // Calculate plan execution metrics
                let completed_nodes = plan.nodes.iter()
                    .filter(|n| n.estimated_completion > 0.8)
                    .count() as f64;
                let total_nodes = plan.nodes.len() as f64;
                let execution_rate = if total_nodes > 0.0 {
                    completed_nodes / total_nodes
                } else {
                    0.0
                };

                // Adjust dimensional positions based on plan execution
                let adjustment_factor = 0.1 * execution_rate;
                
                // Emergence increases with successful plan execution
                // Calculate adjustments as whole numbers
                let emergence_adj = (adjustment_factor * 10.0) as i32;
                let coherence_adj = (adjustment_factor * 8.0) as i32;
                let resilience_adj = (adjustment_factor * 5.0) as i32;
                let intelligence_adj = (adjustment_factor * 7.0) as i32;
                let efficiency_adj = (adjustment_factor * 6.0) as i32;
                let integration_adj = (adjustment_factor * 9.0) as i32;

                // Apply adjustments and clamp to 0-100 range
                cell.dimensional_position.coherence = (cell.dimensional_position.coherence + coherence_adj as f64).clamp(0.0, 100.0);
                cell.dimensional_position.resilience = (cell.dimensional_position.resilience + resilience_adj as f64).clamp(0.0, 100.0);
                cell.dimensional_position.intelligence = (cell.dimensional_position.intelligence + intelligence_adj as f64).clamp(0.0, 100.0);
                cell.dimensional_position.efficiency = (cell.dimensional_position.efficiency + efficiency_adj as f64).clamp(0.0, 100.0);
                cell.dimensional_position.integration = (cell.dimensional_position.integration + integration_adj as f64).clamp(0.0, 100.0);
                
                println!("Cell {} dimensional audit:", cell_id);
                println!("  Plan execution rate: {:.1}%", execution_rate * 100.0);
                println!("  Emergence: {:.1}", cell.dimensional_position.emergence);
                println!("  Coherence: {:.1}", cell.dimensional_position.coherence);
                println!("  Resilience: {:.1}", cell.dimensional_position.resilience);
                println!("  Intelligence: {:.1}", cell.dimensional_position.intelligence);
                println!("  Efficiency: {:.1}", cell.dimensional_position.efficiency);
                println!("  Integration: {:.1}", cell.dimensional_position.integration);
            }
        }
        
        println!("[{}] Dimensional position audit completed", 
            chrono::Local::now().format("%H:%M:%S"));
            
        Ok(())
    }

    fn get_statistics(&self) -> ColonyStatistics {
        let mut stats = ColonyStatistics {
            total_cells: self.cells.len() as u32,
            total_thoughts: 0,
            total_plans: 0,
            successful_plans: 0,
            failed_plans: 0,
            average_cell_energy: 0.0,
            highest_evolution_stage: 0,
            total_cycles: 0,
        };

        for cell in self.cells.values() {
            stats.total_thoughts += cell.thoughts.len() as u32;
            stats.average_cell_energy += cell.energy;
            if let Some(stage) = cell.get_evolution_stage().checked_sub(0) {
                stats.highest_evolution_stage = stats.highest_evolution_stage.max(stage);
            }
        }

        if !self.cells.is_empty() {
            stats.average_cell_energy /= self.cells.len() as f64;
        }

        stats
    }
        pub fn get_cell_batch(&self, cell_ids: &[Uuid]) -> Vec<Cell> {
        cell_ids.iter()
            .filter_map(|id| self.cells.get(id))
            .cloned()
            .collect()
    }

    pub fn update_cell_batch(&mut self, updated_cells: Vec<Cell>) {
        for cell in updated_cells {
            self.cells.insert(cell.id, cell);
        }
    }

    pub async fn process_cell_thoughts(&mut self, cell_id: Uuid) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(cell) = self.cells.get_mut(&cell_id) {
            cell.generate_thought(&self.api_client, &self.mission).await?;
        }
        Ok(())
    }

    pub fn get_cluster_count(&self) -> usize {
        let mut visited = std::collections::HashSet::new();
        let mut clusters = 0;
        
        for &id in self.cells.keys() {
            if !visited.contains(&id) {
                self.explore_cluster(id, &mut visited);
                clusters += 1;
            }
        }
        
        clusters
    }

    fn explore_cluster(&self, cell_id: Uuid, visited: &mut std::collections::HashSet<Uuid>) {
        if !visited.insert(cell_id) {
            return;
        }
        
        if let Some(cell) = self.cells.get(&cell_id) {
            for &neighbor_id in &cell.neighbors {
                self.explore_cluster(neighbor_id, visited);
            }
        }
    }

    pub fn get_max_depth(&self) -> usize {
        self.cells.values()
            .map(|cell| cell.position.z.abs() as usize)
            .max()
            .unwrap_or(0)
    }

    pub fn get_average_energy(&self) -> f64 {
        if self.cells.is_empty() {
            return 0.0;
        }
        let total_energy: f64 = self.cells.values()
            .map(|cell| cell.energy)
            .sum();
        total_energy / self.cells.len() as f64
    }

    pub fn get_total_thoughts(&self) -> usize {
        self.cells.values()
            .map(|cell| cell.thoughts.len())
            .sum()
    }

    pub fn get_total_plans(&self) -> usize {
        self.cells.values()
            .filter(|cell| cell.current_plan.is_some())
            .count()
    }

    pub fn get_mutation_rate(&self) -> f64 {
        let total_cells = self.cells.len() as f64;
        if total_cells == 0.0 {
            return 0.0;
        }
        let evolved_cells = self.cells.values()
            .filter(|cell| cell.energy > 80.0)
            .count() as f64;
        evolved_cells / total_cells
    }
}

fn calculate_dimensional_complement(pos1: &DimensionalPosition, pos2: &DimensionalPosition) -> f64 {
    // Calculate how well two positions complement each other
    // Higher score means their differences tend to balance toward 0
    let emergence_complement = (pos1.emergence + pos2.emergence).abs();
    let coherence_complement = (pos1.coherence + pos2.coherence).abs();
    let resilience_complement = (pos1.resilience + pos2.resilience).abs();
    let intelligence_complement = (pos1.intelligence + pos2.intelligence).abs();
    let efficiency_complement = (pos1.efficiency + pos2.efficiency).abs();
    let integration_complement = (pos1.integration + pos2.integration).abs();

    // Convert to a score where lower differences = higher score
    let total_complement = emergence_complement + coherence_complement + 
                          resilience_complement + intelligence_complement +
                          efficiency_complement + integration_complement;
    
    // Normalize to 0-1 range and invert so higher is better
    let max_possible_diff = 1200.0; // 6 dimensions * 200 (max range of -100 to 100)
    1.0 - (total_complement / max_possible_diff)
}

fn calculate_distance(pos1: &Coordinates, pos2: &Coordinates) -> f64 {
    ((pos2.x - pos1.x).powi(2) + 
     (pos2.y - pos1.y).powi(2) + 
     (pos2.z - pos1.z).powi(2)).sqrt()
}








