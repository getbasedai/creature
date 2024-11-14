use crate::models::types::{CellContext, Coordinates, DimensionalPosition, Plan, RealTimeContext, Thought};
use crate::models::thought_io::{EventInput, EventOutput, ThoughtIO};
use crate::models::constants::MAX_MEMORY_SIZE;
use crate::api::openrouter::OpenRouterClient;
use crate::systems::ltl::{ExtendedNeighborhood, EnhancedCellState, InteractionEffect};
use std::collections::{HashMap, VecDeque};
use std::error::Error;
use chrono::Utc;
use uuid::Uuid;
use rand::Rng;

#[derive(Clone)]
pub struct Cell {
    pub id: Uuid,
    pub position: Coordinates,
    pub thoughts: VecDeque<Thought>,
    pub compressed_memories: Vec<String>,
    pub current_plan: Option<Plan>,
    pub mission_alignment_score: f64,
    pub neighbors: Vec<Uuid>,
    pub energy: f64,
    pub dimensional_position: DimensionalPosition,
    pub dopamine: f64,
    pub research_topics: Vec<String>,
    pub research_depth: u32,
    pub enhanced_state: EnhancedCellState,
    pub neighborhood: ExtendedNeighborhood,
    pub phase: f64,
    pub stability: f64,
    pub influence_radius: f64,
    pub mutation_rate: f64,
    pub lenia_state: f64,
    pub lenia_influence: f64,  // How much Lenia affects this cell
    pub context_influence: f64, // How much real-time context affects this cell
    pub last_context_update: Option<chrono::DateTime<chrono::Utc>>,
    pub context_alignment_score: f64,
}

impl Cell {
    pub fn new(position: Coordinates) -> Self {
        Self {
            id: Uuid::new_v4(),
            position,
            thoughts: VecDeque::new(),
            compressed_memories: Vec::new(),
            current_plan: None,
            mission_alignment_score: 1.0,
            neighbors: Vec::new(),
            energy: 100.0,
            dimensional_position: DimensionalPosition {
                emergence: 50.0,
                coherence: 50.0,
                resilience: 50.0,
                intelligence: 50.0,
                efficiency: 50.0,
                integration: 50.0,
            },
            dopamine: 0.5,
            enhanced_state: EnhancedCellState::new(),
            neighborhood: ExtendedNeighborhood::new(3.0, 12),
            phase: 0.0,
            stability: 1.0,
            influence_radius: 3.0,
            mutation_rate: 1.0, // 100% mutation rate
            research_topics: Vec::new(),
            research_depth: 1,
            lenia_state: 0.0,
            lenia_influence: 0.5,
            context_influence: 0.7,
            last_context_update: None,
            context_alignment_score: 0.5,
        }
    }

    pub async fn update_with_ltl_rules(
        &mut self, 
        api_client: &OpenRouterClient,
        other_cells: &[(Uuid, Coordinates)]
    ) -> Result<(), Box<dyn Error>> {
        self.neighborhood.update_neighbors(&self.position, other_cells);
        let neighbor_states: HashMap<Uuid, EnhancedCellState> = self.neighborhood.neighbors.keys()
            .filter_map(|&id| {
                other_cells.iter()
                    .find(|(other_id, _)| *other_id == id)
                    .map(|_| (id, EnhancedCellState::new()))
            })
            .collect();

        self.enhanced_state.update(&self.neighborhood, &neighbor_states);
        
        let effects = self.calculate_interaction_effects(&neighbor_states);
        self.process_interaction_effects(effects);

        if self.should_generate_thought() {
            let context = self.get_current_focus();
            self.generate_thought(api_client, &context).await?;
        }

        Ok(())
    }

    fn should_generate_thought(&self) -> bool {
        self.enhanced_state.stability > 0.5 && 
        self.enhanced_state.activity_level > 0.3 &&
        rand::random::<f64>() < self.mutation_rate // Use mutation_rate directly for thought generation
    }

    fn calculate_interaction_effects(&self, neighbor_states: &HashMap<Uuid, EnhancedCellState>) -> Vec<InteractionEffect> {
        let mut effects = Vec::new();
        
        // Energy gradient effects
        let avg_neighbor_energy: f64 = neighbor_states.values()
            .map(|state| state.energy)
            .sum::<f64>() / neighbor_states.len() as f64;
        
        let energy_gradient = (avg_neighbor_energy - self.enhanced_state.energy) / 100.0;
        if energy_gradient > 0.5 && self.enhanced_state.stability > 0.7 {
            effects.push(InteractionEffect::EnergyBoost(energy_gradient * 5.0));
        }

        // Phase synchronization
        let phase_alignment = neighbor_states.values()
            .map(|state| (state.phase - self.enhanced_state.phase).cos())
            .sum::<f64>() / neighbor_states.len() as f64;
        
        if phase_alignment > 0.8 {
            effects.push(InteractionEffect::SynchronizationBonus(phase_alignment * 2.0));
        }

        // Reproduction conditions
        if self.enhanced_state.activity_level > 0.7 && self.enhanced_state.energy > 50.0 {
            effects.push(InteractionEffect::SpawnConditionsMet);
        }

        effects
    }

    fn process_interaction_effects(&mut self, effects: Vec<InteractionEffect>) {
        for effect in effects {
            match effect {
                InteractionEffect::EnergyBoost(amount) => {
                    self.enhanced_state.energy += amount;
                    self.energy += amount;
                },
                InteractionEffect::SynchronizationBonus(bonus) => {
                    self.enhanced_state.stability += bonus * 0.1;
                    self.dopamine += bonus * 0.05;
                },
                InteractionEffect::SpawnConditionsMet => {
                    self.enhanced_state.energy *= 0.7;
                    self.energy *= 0.7;
                }
            }
        }
    }

    pub async fn generate_thought(
        &mut self,
        api_client: &OpenRouterClient,
        mission: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // First evaluate dimensional state
        let recent_thoughts: Vec<_> = self.thoughts.iter().rev().take(5).cloned().collect();
        let recent_plans: Vec<_> = self.compressed_memories.iter()
            .filter_map(|m| serde_json::from_str::<Plan>(m).ok())
            .take(5)
            .collect();
            
        let (energy_impact, dopamine_impact) = api_client
            .evaluate_dimensional_state(&self.dimensional_position, &recent_thoughts, &recent_plans)
            .await?;
            
        // Apply impacts
        self.energy = (self.energy + energy_impact).clamp(0.0, 100.0);
        self.dopamine = (self.dopamine + (dopamine_impact - 0.5)).clamp(0.0, 1.0);
        let cell_context = CellContext {
            current_focus: self.get_current_focus(),
            active_research_topics: self.get_active_research(),
            recent_discoveries: self.get_recent_discoveries(),
            collaboration_history: self.get_collaboration_history(),
            performance_metrics: self.get_performance_metrics(),
            evolution_stage: self.get_evolution_stage(),
            energy_level: self.energy,
            dimensional_position: self.dimensional_position.clone(),
            dopamine: self.dopamine,
        };

        // Get real-time context with recent thoughts for better context
        let recent_thought_contents: Vec<String> = self.thoughts.iter()
            .rev()
            .take(3)
            .map(|t| t.content.clone())
            .collect();
        let real_time_context = api_client.gather_real_time_context(Some(recent_thought_contents)).await?;
        
        let (thought_content, relevance_score, factors) = api_client
            .generate_contextual_thought(&cell_context, &real_time_context, mission)
            .await?;

        // Parse dimensional scores from thought content
        for line in thought_content.lines() {
            let line = line.trim();
            if line.starts_with("- EMERGENT_INTELLIGENCE:") {
                if let Some(score_str) = line.split(':').nth(1) {
                    if let Ok(score) = score_str.trim().parse::<f64>() {
                        self.dimensional_position.emergence = score.clamp(-100.0, 100.0);
                    }
                }
            } else if line.starts_with("- RESOURCE_EFFICIENCY:") {
                if let Some(score_str) = line.split(':').nth(1) {
                    if let Ok(score) = score_str.trim().parse::<f64>() {
                        self.dimensional_position.efficiency = score.clamp(-100.0, 100.0);
                    }
                }
            } else if line.starts_with("- NETWORK_COHERENCE:") {
                if let Some(score_str) = line.split(':').nth(1) {
                    if let Ok(score) = score_str.trim().parse::<f64>() {
                        self.dimensional_position.coherence = score.clamp(-100.0, 100.0);
                    }
                }
            } else if line.starts_with("- GOAL_ALIGNMENT:") {
                if let Some(score_str) = line.split(':').nth(1) {
                    if let Ok(score) = score_str.trim().parse::<f64>() {
                        self.dimensional_position.intelligence = score.clamp(-100.0, 100.0);
                    }
                }
            } else if line.starts_with("- TEMPORAL_RESILIENCE:") {
                if let Some(score_str) = line.split(':').nth(1) {
                    if let Ok(score) = score_str.trim().parse::<f64>() {
                        self.dimensional_position.resilience = score.clamp(-100.0, 100.0);
                    }
                }
            } else if line.starts_with("- DIMENSIONAL_INTEGRATION:") {
                if let Some(score_str) = line.split(':').nth(1) {
                    if let Ok(score) = score_str.trim().parse::<f64>() {
                        self.dimensional_position.integration = score.clamp(-100.0, 100.0);
                    }
                }
            }
        }

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut connections = Vec::new();
        
        let input = EventInput {
            id: Uuid::new_v4(),
            event_type: "THOUGHT_GENERATION".to_string(),
            description: thought_content.clone(),
            probability: relevance_score,
            timeframe: "immediate".to_string(),
            requirements: factors.clone(),
        };
        inputs.push(input);

        let output = EventOutput {
            id: Uuid::new_v4(),
            effect_type: "SYSTEM_UPDATE".to_string(), 
            description: "Update system based on thought".to_string(),
            impact_score: relevance_score,
            dependencies: vec![],
            cascading_effects: factors,
        };
        outputs.push(output);

        connections.push((inputs[0].id, outputs[0].id));

        let thought_io = ThoughtIO {
            inputs,
            outputs,
            connection_graph: connections,
        };

        // Convert ThoughtIO into Vec<String> containing relevant information
        let mut factors = Vec::new();
        factors.extend(thought_io.inputs.iter().map(|input| input.description.clone()));
        factors.extend(thought_io.outputs.iter().map(|output| output.description.clone()));
        
        // Filter out restricted terms
        let filtered_content = thought_content.replace("quantum", "advanced");
        
        let thought = Thought {
            id: Uuid::new_v4(),
            content: filtered_content,
            timestamp: Utc::now(),
            relevance_score,
            context_tags: self.generate_context_tags(&cell_context),
            real_time_factors: factors,
            confidence_score: self.calculate_confidence_score(&real_time_context),
        };
        println!("\nGenerated Thought:");
        println!("════════════════════════════════════════════════════════════════════");
        println!("ID: {}", thought.id);
        println!("Content:\n{}", thought.content);
        println!("Relevance Score: {:.2}", thought.relevance_score);
        println!("Confidence Score: {:.2}", thought.confidence_score);
        println!("Context Tags: {}", thought.context_tags.join(", "));
        println!("Real-time Factors:");
        for factor in &thought.real_time_factors {
            println!("  - {}", factor);
        }
        println!("════════════════════════════════════════════════════════════════════\n");

        if let Some(ref mut plan) = self.current_plan {
            if plan.summary.trim().is_empty() {
                plan.summary = format!("Plan based on thought: {}", thought_content);
            }
        }

        // Log thought to file before adding to memory
        if let Err(e) = crate::utils::logging::log_thought_to_file(&self.id, &thought) {
            eprintln!("Error logging thought to file: {}", e);
        }
        
        self.thoughts.push_back(thought);
        self.check_and_compress_memories(api_client).await?;
        self.update_focus_based_on_context(&real_time_context);
        
        // Save state after thought generation
        // State saving handled by Colony
        
        Ok(())
    }

    pub async fn check_and_compress_memories(&mut self, api_client: &OpenRouterClient) -> Result<(), Box<dyn std::error::Error>> {
        let total_size: usize = self.thoughts.iter().map(|t| t.content.len()).sum();

        if total_size > MAX_MEMORY_SIZE {
            let thoughts_to_compress: Vec<String> = self.thoughts
                .drain(..self.thoughts.len() / 2)
                .map(|t| t.content)
                .collect();

            let compressed = api_client.compress_memories(&thoughts_to_compress).await?;
            self.compressed_memories.push(compressed);
        }

        Ok(())
    }

    pub fn get_current_focus(&self) -> String {
        self.current_plan
            .as_ref()
            .map(|p| p.summary.clone())
            .unwrap_or_else(|| "Exploring new opportunities".to_string())
    }

    pub fn get_active_research(&self) -> Vec<String> {
        self.thoughts
            .iter()
            .take(5)
            .map(|t| t.content.clone())
            .collect()
    }

    pub fn get_recent_discoveries(&self) -> Vec<String> {
        self.thoughts
            .iter()
            .filter(|t| t.relevance_score > 0.8)
            .take(3)
            .map(|t| t.content.clone())
            .collect()
    }

    pub fn get_collaboration_history(&self) -> Vec<String> {
        vec!["Previous collaborations".to_string()]
    }

    pub fn get_performance_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("energy_efficiency".to_string(), self.energy);
        metrics.insert("mission_alignment".to_string(), self.mission_alignment_score);
        metrics
    }

    pub fn get_evolution_stage(&self) -> u32 {
        (self.thoughts.len() as f64 / 10.0).floor() as u32 + 1
    }

    pub fn log_current_plan(&self) {
        use crate::utils::logging::*;

        if let Some(ref plan) = self.current_plan {
            log_section_header("═══════════════ STRATEGIC PLAN ANALYSIS ═══════════════");
            
            // Plan Overview
            println!("╔════════════════════════ PLAN OVERVIEW ════════════════════════╗");
            log_simple_metric("Plan ID", plan.id);
            log_simple_metric("Created", plan.created_at.format("%Y-%m-%d %H:%M:%S"));
            log_simple_metric("Status", format!("{:?}", plan.status));
            log_simple_metric("Score", format!("{:.2}", plan.score));
            println!("╚═══════════════════════════════════════════════════════════════╝\n");

            // Plan Summary
            println!("╔═════════════════════════ EXECUTIVE SUMMARY ══════════════════════╗");
            for line in plan.summary.lines() {
                println!("║ {:<69} ║", line);
            }
            println!("╚═══════════════════════════════════════════════════════════════════╝\n");

            // Plan Components
            if !plan.nodes.is_empty() {
                println!("╔═════════════════════════ PLAN COMPONENTS ═══════════════════════╗");
                for (idx, node) in plan.nodes.iter().enumerate() {
                    println!("║");
                    println!("║ [{:02}] {}", idx + 1, "═".repeat(63));
                    println!("║     Title: {}", node.title);
                    println!("║     Status: {:?}", node.status);
                    println!("║     Progress: [{}{}] {:.1}%",
                        "█".repeat((node.estimated_completion * 20.0) as usize),
                        "░".repeat(20 - (node.estimated_completion * 20.0) as usize),
                        node.estimated_completion * 100.0
                    );
                    println!("║");
                    println!("║     Description:");
                    for line in node.description.lines() {
                        println!("║       {}", line);
                    }
                }
                println!("╚═══════════════════════════════════════════════════════════════════╝\n");
            }

            // Supporting Thoughts
            if !plan.thoughts.is_empty() {
                println!("╔════════════════════ SUPPORTING INTELLIGENCE ═══════════════════╗");
                for (idx, thought) in plan.thoughts.iter().enumerate() {
                    println!("║");
                    println!("║ Thought {}: [Relevance: {:.2}] [Confidence: {:.2}]", 
                        idx + 1, 
                        thought.relevance_score,
                        thought.confidence_score
                    );
                    println!("║ ─────────────────────────────────────────────────────────────");
                    for line in thought.content.lines() {
                        println!("║   {}", line);
                    }
                    if !thought.context_tags.is_empty() {
                        println!("║   Tags: {}", thought.context_tags.join(", "));
                    }
                    if !thought.real_time_factors.is_empty() {
                        println!("║   Factors: {}", thought.real_time_factors.join(", "));
                    }
                }
                println!("╚═══════════════════════════════════════════════════════════════════╝\n");
            }

            // Participating Cells
            if !plan.participating_cells.is_empty() {
                println!("╔═════════════════════ PARTICIPATING CELLS ═══════════════════════╗");
                println!("║ Total Participants: {}", plan.participating_cells.len());
                println!("║");
                for (idx, cell_id) in plan.participating_cells.iter().enumerate() {
                    println!("║ [{:02}] {}", idx + 1, cell_id);
                }
                println!("╚═══════════════════════════════════════════════════════════════════╝");
            }

            log_section_footer();
        } else {
            log_warning("No active plan exists for this cell");
        }
    }

    pub fn generate_context_tags(&self, context: &CellContext) -> Vec<String> {
        vec![
            format!("stage_{}", context.evolution_stage),
            format!("energy_{}", (context.energy_level / 20.0).floor() * 20.0),
            "active".to_string(),
        ]
    }

    pub fn calculate_confidence_score(&self, _context: &RealTimeContext) -> f64 {
        let mut rng = rand::thread_rng();
        0.8 + (rng.gen::<f64>() * 0.2)
    }

    fn update_focus_based_on_context(&mut self, context: &RealTimeContext) {
        // Update last context time
        self.last_context_update = Some(Utc::now());
        
        // Calculate context relevance score
        let mut context_relevance = 0.0;
        let current_focus = self.get_current_focus().to_lowercase();
        
        // Check market trends alignment
        for trend in &context.market_trends {
            if current_focus.contains(&trend.to_lowercase()) {
                context_relevance += 0.2;
            }
        }
        
        // Check technological developments alignment
        for tech in &context.technological_developments {
            if current_focus.contains(&tech.to_lowercase()) {
                context_relevance += 0.3;
            }
        }
        
        // Check current events alignment
        for event in &context.current_events {
            if current_focus.contains(&event.to_lowercase()) {
                context_relevance += 0.2;
            }
        }
        
        // Check user interactions alignment
        for interaction in &context.user_interactions {
            if current_focus.contains(&interaction.to_lowercase()) {
                context_relevance += 0.3;
            }
        }
        
        // Update context alignment score
        self.context_alignment_score = self.context_alignment_score * 0.8 + context_relevance * 0.2;
        
        // Adjust energy based on context alignment
        let energy_adjustment = (self.context_alignment_score - 0.5) * 10.0 * self.context_influence;
        self.energy = (self.energy + energy_adjustment).clamp(0.0, 100.0);
        
        // Adjust mission alignment based on context
        self.mission_alignment_score *= 0.95;
        self.mission_alignment_score += self.context_alignment_score * 0.05;
        
        // Update dimensional position based on context alignment
        let dimension_adjustment = (self.context_alignment_score - 0.5) * 2.0;
        // Calculate adjustments as whole numbers
        let emergence_adj = (dimension_adjustment * 10.0) as i32;
        let coherence_adj = (dimension_adjustment * 10.0) as i32;
        let intelligence_adj = (dimension_adjustment * 15.0) as i32;
        let efficiency_adj = (dimension_adjustment * 10.0) as i32;
        
        // Apply adjustments and clamp to 0-100 range
        self.dimensional_position.emergence = (self.dimensional_position.emergence + emergence_adj as f64).clamp(0.0, 100.0);
        self.dimensional_position.coherence = (self.dimensional_position.coherence + coherence_adj as f64).clamp(0.0, 100.0);
        self.dimensional_position.intelligence = (self.dimensional_position.intelligence + intelligence_adj as f64).clamp(0.0, 100.0);
        self.dimensional_position.efficiency = (self.dimensional_position.efficiency + efficiency_adj as f64).clamp(0.0, 100.0);
        
        // Adjust dopamine based on context alignment
        self.dopamine = self.dopamine * 0.9 + self.context_alignment_score * 0.1;
    }
}
