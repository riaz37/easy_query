#!/usr/bin/env python3
"""
Example demonstrating the Report Generation functionality through Navigation Tool.
Shows the expected input/output format for report generation as buttons on database-query page.
"""

import asyncio
import json
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIConfig
from tools.navigation_tool import create_navigation_tool


async def demonstrate_report_generation():
    """Demonstrate the report generation functionality with example scenarios."""
    print("📊 Report Generation via Navigation Tool Demonstration")
    print("=" * 60)
    
    # Create RTVI processor
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
    
    # Create the navigation tool
    tool = create_navigation_tool(rtvi)
    
    # Example 1: Generate report
    print("\n📋 Example 1: Generate report")
    print("User input: 'I want to generate a financial report'")
    
    result1 = await tool.execute(
        user_id="test_user",
        target="Generate a financial report for Q1 2024",
        action_type="generate_report",
        context="financial data analysis"
    )
    
    print(f"Tool output: {result1}")
    print(f"Structured output that would be sent to frontend:")
    print(json.dumps({
        "Action_type": "navigation",
        "param": "clicked,name,report_query",
        "value": "true,report generation,Generate a financial report for Q1 2024",
        "page": "database-query",
        "previous_page": None,
        "interaction_type": "generate_report",
        "clicked": True,
        "element_name": "report generation",
        "report_query": "Generate a financial report for Q1 2024",
        "context": "financial data analysis"
    }, indent=2))
    
    # Example 2: View report
    print("\n📋 Example 2: View report")
    print("User input: 'Show me the sales report'")
    
    result2 = await tool.execute(
        user_id="test_user",
        target="Show me the sales report from last month",
        action_type="view_report",
        context="sales data"
    )
    
    print(f"Tool output: {result2}")
    print(f"Structured output that would be sent to frontend:")
    print(json.dumps({
        "Action_type": "navigation",
        "param": "clicked,name,report_request",
        "value": "true,view report,Show me the sales report from last month",
        "page": "database-query",
        "previous_page": None,
        "interaction_type": "view_report",
        "clicked": True,
        "element_name": "view report",
        "report_request": "Show me the sales report from last month",
        "context": "sales data"
    }, indent=2))
    
    # Example 3: Employee performance report
    print("\n📋 Example 3: Employee performance report")
    print("User input: 'Create an employee performance report'")
    
    result3 = await tool.execute(
        user_id="test_user",
        target="Create an employee performance report for the current quarter",
        action_type="generate_report",
        context="HR performance metrics"
    )
    
    print(f"Tool output: {result3}")
    print(f"Structured output that would be sent to frontend:")
    print(json.dumps({
        "Action_type": "navigation",
        "param": "clicked,name,report_query",
        "value": "true,report generation,Create an employee performance report for the current quarter",
        "page": "database-query",
        "previous_page": None,
        "interaction_type": "generate_report",
        "clicked": True,
        "element_name": "report generation",
        "report_query": "Create an employee performance report for the current quarter",
        "context": "HR performance metrics"
    }, indent=2))
    
    print("\n✅ Demonstration completed!")
    print("\n📝 Summary of the new report generation approach:")
    print("- Function name: navigate_page")
    print("- Action types: 'generate_report' or 'view_report'")
    print("- Page: database-query (with buttons: 'view report', 'report generation')")
    print("- Input parameters: user_id, target, action_type, context")
    print("- Output structure: Action_type='navigation', interaction_type='generate_report' or 'view_report'")
    print("- Value: The user's report request as provided")


if __name__ == "__main__":
    asyncio.run(demonstrate_report_generation())
