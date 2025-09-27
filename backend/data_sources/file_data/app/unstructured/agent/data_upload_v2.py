#!/usr/bin/env python3
"""
Document Processing Pipeline
Orchestrates document saving, embedding generation, sub-intent generation, and intent mapping in sequence.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Import the functions from your modules
from data_sources.file_data.app.unstructured.agent.data_storage_v2 import data_save
from data_sources.file_data.app.unstructured.agent.embedding import embed_gen
from data_sources.file_data.app.unstructured.agent.sub_intent import sub_intent_gen, SubIntentConfig, SubIntentManager
from data_sources.file_data.app.unstructured.agent.intent_chunk import intent_mapping_main, IntentMappingConfig, IntentMappingManager
from data_sources.file_data.app.unstructured.agent.config_loader import get_database_config, DatabaseConfig

# Load environment variables
load_dotenv(override=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentPipelineOrchestrator:
    """Orchestrates the complete document processing, embedding, sub-intent, and intent mapping pipeline."""
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process_document(
        self,
        file_path: str,
        # Document processing parameters
        max_pages_per_chunk: int = 5,
        boundary_sentences: int = 3,
        boundary_table_rows: int = 3,
        target_pages_per_chunk: int = 3,
        overlap_pages: int = 1,
        min_pages_per_chunk: int = 1,
        respect_boundaries: bool = True,
        max_workers: int = 4,
        # Embedding parameters
        batch_size: int = 4,
        delay_between_requests: int = 5,
        max_retries: int = 3,
        retry_delay: float = 20,
        # Sub-intent parameters
        similarity_threshold: float = 0.75,
        # Intent mapping parameters
        intent_similarity_threshold: float = 0.70,
        top_n_candidates: int = 5,
        intent_batch_size: int = 20,
        intent_delay: float = 1.0,
        # New parameters
        file_description: Optional[str] = None,
        table_name: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a document through the complete pipeline.
        
        Args:
            file_path: Path to the document to process
            Document processing parameters (see data_save function)
            Embedding parameters (see embed_gen function)
            Sub-intent parameters
            Intent mapping parameters
            
        Returns:
            Dict containing pipeline results and status
        """
        pipeline_result = {
            "success": False,
            "file_path": file_path,
            "document_processing": {"success": False, "error": None},
            "embedding_generation": {"success": False, "error": None},
            "sub_intent_generation": {"success": False, "error": None},
            "intent_mapping": {"success": False, "error": None},
            "error": None
        }
        
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            self.logger.info(f"🚀 Starting complete pipeline for: {file_path}")
            
            # Step 1: Document Processing and Saving
            self.logger.info("📄 Step 1: Processing and saving document...")
            try:
                data_save(
                    file_path=file_path,
                    max_pages_per_chunk=max_pages_per_chunk,
                    boundary_sentences=boundary_sentences,
                    boundary_table_rows=boundary_table_rows,
                    target_pages_per_chunk=target_pages_per_chunk,
                    overlap_pages=overlap_pages,
                    min_pages_per_chunk=min_pages_per_chunk,
                    respect_boundaries=respect_boundaries,
                    max_workers=max_workers,
                    file_description=file_description,
                    table_name=table_name,
                    user_id=user_id
                )
                pipeline_result["document_processing"]["success"] = True
                self.logger.info("✅ Document processing completed successfully!")
                
            except Exception as e:
                error_msg = f"Document processing failed: {str(e)}"
                self.logger.error(error_msg)
                pipeline_result["document_processing"]["error"] = error_msg
                pipeline_result["error"] = error_msg
                return pipeline_result
            
            # Step 2: Embedding Generation
            self.logger.info("🧠 Step 2: Generating embeddings...")
            try:
                asyncio.run(embed_gen(
                    batch_size=batch_size,
                    delay_between_requests=delay_between_requests,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    user_id=user_id
                ))
                pipeline_result["embedding_generation"]["success"] = True
                self.logger.info("✅ Embedding generation completed successfully!")
                
            except Exception as e:
                error_msg = f"Embedding generation failed: {str(e)}"
                self.logger.error(error_msg)
                pipeline_result["embedding_generation"]["error"] = error_msg
                pipeline_result["error"] = error_msg
                return pipeline_result
            
            # Step 3: Sub Intent Generation
            self.logger.info("🔍 Step 3: Generating sub-intents...")
            try:
                # Get gemini API key from environment
                gemini_apikey = os.getenv('google_api_key')
                if not gemini_apikey:
                    raise ValueError("GEMINI_API_KEY environment variable not set")
                
                # Create sub-intent configuration
                sub_intent_config = SubIntentConfig(
                    api_key=gemini_apikey,
                    similarity_threshold=similarity_threshold,
                    batch_size=batch_size,
                    delay_between_requests=delay_between_requests,
                    max_retries=max_retries,
                    retry_delay=retry_delay
                )
                
                # Load database config from user configuration
                db_config = get_database_config(user_id)
                manager = SubIntentManager(sub_intent_config, db_config)
                
                # Run sub-intent generation asynchronously
                asyncio.run(manager.process_all_files())
                
                pipeline_result["sub_intent_generation"]["success"] = True
                self.logger.info("✅ Sub-intent generation completed successfully!")
                
            except Exception as e:
                error_msg = f"Sub-intent generation failed: {str(e)}"
                self.logger.error(error_msg)
                pipeline_result["sub_intent_generation"]["error"] = error_msg
                pipeline_result["error"] = error_msg
                return pipeline_result
            
            # Step 4: Intent Mapping
            self.logger.info("🎯 Step 4: Mapping intents...")
            try:
                # Get gemini API key from environment (already retrieved above)
                gemini_apikey = os.getenv('google_api_key')
                if not gemini_apikey:
                    raise ValueError("GEMINI_API_KEY environment variable not set")
                
                # Create intent mapping configuration
                intent_config = IntentMappingConfig(
                    api_key=gemini_apikey,
                    similarity_threshold=intent_similarity_threshold,
                    top_n_candidates=top_n_candidates,
                    batch_size=intent_batch_size,
                    delay_between_requests=intent_delay,
                    max_retries=max_retries,
                    retry_delay=retry_delay
                )
                
                # Load database config from user configuration
                db_config = get_database_config(user_id)
                intent_manager = IntentMappingManager(intent_config, db_config)
                
                # Run intent mapping asynchronously
                asyncio.run(intent_manager.process_all_chunks())
                
                pipeline_result["intent_mapping"]["success"] = True
                self.logger.info("✅ Intent mapping completed successfully!")
                
            except Exception as e:
                error_msg = f"Intent mapping failed: {str(e)}"
                self.logger.error(error_msg)
                pipeline_result["intent_mapping"]["error"] = error_msg
                pipeline_result["error"] = error_msg
                return pipeline_result
            
            # Pipeline completed successfully
            pipeline_result["success"] = True
            self.logger.info("🎉 Complete pipeline finished successfully!")
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            pipeline_result["error"] = error_msg
        
        return pipeline_result
    
    def process_multiple_documents(
        self,
        file_paths: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process multiple documents through the pipeline.
        
        Args:
            file_paths: List of file paths to process
            **kwargs: Parameters to pass to process_document
            
        Returns:
            Dict containing results for all processed documents
        """
        results = {
            "total_files": len(file_paths),
            "successful": 0,
            "failed": 0,
            "results": []
        }
        
        self.logger.info(f"🚀 Starting batch processing for {len(file_paths)} files...")
        
        for i, file_path in enumerate(file_paths, 1):
            self.logger.info(f"📄 Processing file {i}/{len(file_paths)}: {file_path}")
            
            result = self.process_document(file_path, **kwargs)
            results["results"].append(result)
            
            if result["success"]:
                results["successful"] += 1
                self.logger.info(f"✅ File {i} completed successfully")
            else:
                results["failed"] += 1
                self.logger.error(f"❌ File {i} failed: {result['error']}")
        
        self.logger.info(f"🎉 Batch processing completed: {results['successful']}/{results['total_files']} successful")
        return results


def process_single_document(
    file_path: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to process a single document.
    
    Args:
        file_path: Path to the document to process
        **kwargs: Additional parameters for processing
        
    Returns:
        Processing result dictionary
    """
    orchestrator = DocumentPipelineOrchestrator()
    return orchestrator.process_document(file_path, **kwargs)


def process_directory(
    directory_path: str,
    file_extensions: List[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Process all documents in a directory.
    
    Args:
        directory_path: Path to directory containing documents
        file_extensions: List of file extensions to process
        **kwargs: Additional parameters for processing
        
    Returns:
        Batch processing result dictionary
    """
    if file_extensions is None:
        file_extensions = ['.pdf', '.docx', '.txt']
    
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    # Find all files with specified extensions
    file_paths = []
    for ext in file_extensions:
        file_paths.extend(directory.glob(f"*{ext}"))
    
    file_paths = [str(path) for path in file_paths]
    
    if not file_paths:
        logger.warning(f"No files found with extensions {file_extensions} in {directory_path}")
        return {"total_files": 0, "successful": 0, "failed": 0, "results": []}
    
    orchestrator = DocumentPipelineOrchestrator()
    return orchestrator.process_multiple_documents(file_paths, **kwargs)


def run_individual_stage(
    stage: str,
    file_path: Optional[str] = None,
    # Document processing parameters
    max_pages_per_chunk: int = 5,
    boundary_sentences: int = 3,
    boundary_table_rows: int = 3,
    target_pages_per_chunk: int = 3,
    overlap_pages: int = 1,
    min_pages_per_chunk: int = 1,
    respect_boundaries: bool = True,
    max_workers: int = 4,
    # Embedding parameters
    batch_size: int = 20,
    delay_between_requests: int = 2,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    # Sub-intent parameters
    similarity_threshold: float = 0.75,
    # Intent mapping parameters
    intent_similarity_threshold: float = 0.70,
    top_n_candidates: int = 5,
    intent_batch_size: int = 20,
    intent_delay: float = 1.0,
    # New parameters
    file_description: Optional[str] = None,
    table_name: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a specific stage of the document processing pipeline independently.
    
    Args:
        stage: Stage to run - 'store', 'embed', 'sub_intent', or 'intent_mapping'
        file_path: Required for 'store' stage, optional for others
        Other parameters: Same as in the main pipeline
        
    Returns:
        Dict containing stage results and status
    """
    stage_result = {
        "success": False,
        "stage": stage,
        "file_path": file_path,
        "error": None
    }
    
    try:
        if stage.lower() == 'store':
            # Document Processing and Saving Stage
            if not file_path:
                raise ValueError("file_path is required for 'store' stage")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            logger.info(f"📄 Running STORE stage for: {file_path}")
            logger.info(f"DEBUG: Passing user_id={user_id} to data_save")
            
            data_save(
                file_path=file_path,
                max_pages_per_chunk=max_pages_per_chunk,
                boundary_sentences=boundary_sentences,
                boundary_table_rows=boundary_table_rows,
                target_pages_per_chunk=target_pages_per_chunk,
                overlap_pages=overlap_pages,
                min_pages_per_chunk=min_pages_per_chunk,
                respect_boundaries=respect_boundaries,
                max_workers=max_workers,
                file_description=file_description,
                table_name=table_name,
                user_id=user_id
            )
            logger.info("✅ Document storage completed successfully!")
            
        elif stage.lower() == 'embed':
            # Embedding Generation Stage
            logger.info("🧠 Running EMBED stage...")
            
            asyncio.run(embed_gen(
                batch_size=batch_size,
                delay_between_requests=delay_between_requests,
                max_retries=max_retries,
                retry_delay=retry_delay,
                user_id=user_id
            ))
            logger.info("✅ Embedding generation completed successfully!")
            
        elif stage.lower() == 'sub_intent':
            # Sub Intent Generation Stage
            logger.info("🔍 Running SUB_INTENT stage...")
            
            # Get gemini API key from environment
            gemini_apikey = os.getenv('google_api_key')
            if not gemini_apikey:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            # Create sub-intent configuration
            sub_intent_config = SubIntentConfig(
                api_key=gemini_apikey,
                similarity_threshold=similarity_threshold,
                batch_size=batch_size,
                delay_between_requests=delay_between_requests,
                max_retries=max_retries,
                retry_delay=retry_delay
            )
            
            # Load database config from API or environment variables
            db_config = get_database_config(user_id)
            manager = SubIntentManager(sub_intent_config, db_config)
            
            # Run sub-intent generation
            asyncio.run(manager.process_all_files())
            logger.info("✅ Sub-intent generation completed successfully!")
            
        elif stage.lower() == 'intent_mapping':
            # Intent Mapping Stage
            logger.info("🎯 Running INTENT_MAPPING stage...")
            
            # Get gemini API key from environment
            gemini_apikey = os.getenv('google_api_key')
            if not gemini_apikey:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            # Create intent mapping configuration
            intent_config = IntentMappingConfig(
                api_key=gemini_apikey,
                similarity_threshold=intent_similarity_threshold,
                top_n_candidates=top_n_candidates,
                batch_size=intent_batch_size,
                delay_between_requests=intent_delay,
                max_retries=max_retries,
                retry_delay=retry_delay
            )
            
            # Load database config from API or environment variables
            db_config = get_database_config(user_id)
            intent_manager = IntentMappingManager(intent_config, db_config)
            
            # Run intent mapping
            asyncio.run(intent_manager.process_all_chunks())
            logger.info("✅ Intent mapping completed successfully!")
            
        else:
            raise ValueError(f"Invalid stage: '{stage}'. Must be 'store', 'embed', 'sub_intent', or 'intent_mapping'")
        
        stage_result["success"] = True
        
    except Exception as e:
        error_msg = f"Stage '{stage}' failed: {str(e)}"
        logger.error(error_msg)
        stage_result["error"] = error_msg
    
    return stage_result


def run_store_only(file_path: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to run only the document storage stage."""
    return run_individual_stage('store', file_path=file_path, **kwargs)


def run_embed_only(**kwargs) -> Dict[str, Any]:
    """Convenience function to run only the embedding generation stage."""
    return run_individual_stage('embed', **kwargs)


def run_sub_intent_only(**kwargs) -> Dict[str, Any]:
    """Convenience function to run only the sub-intent generation stage."""
    return run_individual_stage('sub_intent', **kwargs)


def run_intent_mapping_only(**kwargs) -> Dict[str, Any]:
    """Convenience function to run only the intent mapping stage."""
    return run_individual_stage('intent_mapping', **kwargs)


def run_partial_pipeline(
    stages: List[str],
    file_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run multiple specific stages of the pipeline.
    
    Args:
        stages: List of stages to run e.g., ['store', 'embed'] or ['embed', 'sub_intent', 'intent_mapping']
        file_path: Required if 'store' is in stages
        **kwargs: Parameters for the stages
        
    Returns:
        Dict containing results for all requested stages
    """
    pipeline_result = {
        "success": False,
        "stages_requested": stages,
        "file_path": file_path,
        "stage_results": {},
        "error": None
    }
    
    logger.info(f"🚀 Starting partial pipeline with stages: {stages}")
    
    try:
        # Validate stages
        valid_stages = ['store', 'embed', 'sub_intent', 'intent_mapping']
        for stage in stages:
            if stage.lower() not in valid_stages:
                raise ValueError(f"Invalid stage: '{stage}'. Valid stages: {valid_stages}")
        
        # Check if file_path is required
        if 'store' in [s.lower() for s in stages] and not file_path:
            raise ValueError("file_path is required when 'store' stage is included")
        
        # Process each stage
        all_successful = True
        for stage in stages:
            stage_result = run_individual_stage(stage, file_path=file_path, **kwargs)
            pipeline_result["stage_results"][stage] = stage_result
            
            if not stage_result["success"]:
                all_successful = False
                pipeline_result["error"] = stage_result["error"]
                logger.error(f"❌ Stage '{stage}' failed, stopping pipeline")
                break
            
            logger.info(f"✅ Stage '{stage}' completed successfully")
        
        pipeline_result["success"] = all_successful
        
        if all_successful:
            logger.info("🎉 Partial pipeline completed successfully!")
        
    except Exception as e:
        error_msg = f"Partial pipeline failed: {str(e)}"
        logger.error(error_msg)
        pipeline_result["error"] = error_msg
    
    return pipeline_result


async def standalone_sub_intent_gen(
    batch_size: int = 5,
    delay_between_requests: float = 1.5,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    similarity_threshold: float = 0.75,
    user_id: Optional[str] = None
):
    """
    Standalone sub-intent generation function for direct execution.
    
    Args:
        batch_size: Number of files to process in each batch
        delay_between_requests: Delay between API requests
        max_retries: Maximum number of retries for failed requests
        retry_delay: Delay before retrying failed requests
        similarity_threshold: Similarity threshold for sub-intent matching
    """
    print("🚀 Starting Sub-Intent Processing System")
    print("="*50)
    
    try:
        # Get gemini API key from environment
        gemini_apikey = os.getenv('google_api_key')
        if not gemini_apikey:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Configuration
        sub_intent_config = SubIntentConfig(
            api_key=gemini_apikey,
            similarity_threshold=similarity_threshold,
            batch_size=batch_size,
            delay_between_requests=delay_between_requests,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        # Load database config from API or environment variables
        db_config = get_database_config(user_id)
        
        # Initialize manager
        manager = SubIntentManager(sub_intent_config, db_config)
        
        # Process all files
        await manager.process_all_files()
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        if 'manager' in locals():
            manager.display_progress()
        
    except Exception as e:
        logger.error(f"Fatal error during processing: {e}")
        print(f"\n❌ Fatal error: {e}")
        
    finally:
        print("\n👋 Sub-Intent Processing System shutting down...")


async def standalone_intent_mapping(
    batch_size: int = 20,
    delay_between_requests: float = 1.0,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    similarity_threshold: float = 0.70,
    top_n_candidates: int = 5,
    user_id: Optional[str] = None
):
    """
    Standalone intent mapping function for direct execution.
    
    Args:
        batch_size: Number of chunks to process in each batch
        delay_between_requests: Delay between API requests
        max_retries: Maximum number of retries for failed requests
        retry_delay: Delay before retrying failed requests
        similarity_threshold: Similarity threshold for intent matching
        top_n_candidates: Number of top candidates to consider
    """
    print("🚀 Starting Intent Mapping System")
    print("="*50)
    
    try:
        # Get gemini API key from environment
        gemini_apikey = os.getenv('google_api_key')
        if not gemini_apikey:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Configuration
        intent_config = IntentMappingConfig(
            api_key=gemini_apikey,
            similarity_threshold=similarity_threshold,
            top_n_candidates=top_n_candidates,
            batch_size=batch_size,
            delay_between_requests=delay_between_requests,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        # Load database config from API or environment variables
        db_config = get_database_config(user_id)
        
        # Initialize manager
        manager = IntentMappingManager(intent_config, db_config)
        
        # Process all chunks
        await manager.process_all_chunks()
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        if 'manager' in locals():
            manager.display_progress()
        
    except Exception as e:
        logger.error(f"Fatal error during processing: {e}")
        print(f"\n❌ Fatal error: {e}")
        
    finally:
        print("\n👋 Intent Mapping System shutting down...")


if __name__ == "__main__":
    # Example usage - uncomment the section you want to run
    
    # # Process a single document through complete pipeline (all 4 stages)
    # file_path = r"/Users/nilab/Desktop/projects/Knowladge-Base/Agronochain Tech Doc.pdf"
    # result = process_single_document(
    #     file_path=file_path,
    #     # Document processing parameters
    #     max_pages_per_chunk=5,
    #     target_pages_per_chunk=3,
    #     overlap_pages=1,
    #     # Embedding parameters
    #     batch_size=20,
    #     delay_between_requests=2,
    #     max_retries=3,
    #     # Sub-intent parameters
    #     similarity_threshold=0.75,
    #     # Intent mapping parameters
    #     intent_similarity_threshold=0.70,
    #     top_n_candidates=5,
    #     intent_batch_size=20,
    #     intent_delay=1.0
    # )
    # 
    # if result["success"]:
    #     print("🎉 Complete pipeline finished successfully!")
    #     print(f"Document processing: {'✅' if result['document_processing']['success'] else '❌'}")
    #     print(f"Embedding generation: {'✅' if result['embedding_generation']['success'] else '❌'}")
    #     print(f"Sub-intent generation: {'✅' if result['sub_intent_generation']['success'] else '❌'}")
    #     print(f"Intent mapping: {'✅' if result['intent_mapping']['success'] else '❌'}")
    # else:
    #     print(f"❌ Pipeline failed: {result['error']}")
    
    # # Process entire directory
    # directory_result = process_directory(
    #     directory_path="/path/to/documents/",
    #     file_extensions=['.pdf', '.docx'],
    #     batch_size=15,
    #     max_pages_per_chunk=8,
    #     similarity_threshold=0.8,
    #     intent_similarity_threshold=0.70
    # )
    
    # Individual stage examples
    # Example 1: Run only document storage
    # file_path = "/path/to/document.pdf"
    # store_result = run_store_only(
    #     file_path=file_path,
    #     max_pages_per_chunk=5,
    #     target_pages_per_chunk=3
    # )
    # print(f"Store stage: {'✅' if store_result['success'] else '❌'}")
    
    # Example 2: Run only embedding generation
    # embed_result = run_embed_only(
    #     batch_size=15,
    #     delay_between_requests=1.5
    # )
    # print(f"Embed stage: {'✅' if embed_result['success'] else '❌'}")
    
    # Example 3: Run only sub-intent generation
    # sub_intent_result = run_sub_intent_only(
    #     similarity_threshold=0.75,
    #     batch_size=10
    # )
    # print(f"Sub-intent stage: {'✅' if sub_intent_result['success'] else '❌'}")
    
    # Example 4: Run only intent mapping
    intent_mapping_result = run_intent_mapping_only(
        intent_similarity_threshold=0.70,
        top_n_candidates=5,
        intent_batch_size=20,
        intent_delay=1.0
    )
    print(f"Intent mapping stage: {'✅' if intent_mapping_result['success'] else '❌'}")
    
    # # Example 5: Run specific combination of stages
    # partial_result = run_partial_pipeline(
    #     stages=['embed', 'sub_intent', 'intent_mapping'],
    #     batch_size=20,
    #     similarity_threshold=0.75,
    #     intent_similarity_threshold=0.70
    # )
    # print(f"Partial pipeline: {'✅' if partial_result['success'] else '❌'}")
    
    # # Example 6: Standalone sub-intent generation (async)
    # asyncio.run(standalone_sub_intent_gen(
    #     batch_size=5,
    #     delay_between_requests=1.5,
    #     similarity_threshold=0.75
    # ))
    
    # # Example 7: Standalone intent mapping (async)
    # asyncio.run(standalone_intent_mapping(
    #     batch_size=20,
    #     delay_between_requests=1.0,
    #     similarity_threshold=0.70,
    #     top_n_candidates=5
    # ))