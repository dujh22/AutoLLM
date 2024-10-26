# Project Overview

This project automates the periodic processing and prioritization of information from various online sources (e.g., blogs, academic papers, shared content). The architecture is modular and uses multiple agents, each with specific roles in data collection, analysis, and content generation. The automation is designed to run on a daily schedule.

# System Components

## Source Links

The project extracts information from three main types of sources:

	•	Blogs (Source Link 1)
	•	Academic Papers (Source Link 2)
	•	Shared Content (Source Link 3)

Users can also provide additional input to supplement these sources.

## Agents

	1.	Agent1 - Automated Parsing Script Generation
	•	Creates automated scripts to parse and extract information from specified links.
	•	Checks if an existing parser is available in the local directory (spider_lab). If not, generates a new script for the source link.
	2.	Agent2 - Link Information Expansion
	•	Extends the information collected by Agent1 by fetching related links and relevant data for each source link.
	3.	Agent3 - Information Prioritization
	•	Organizes information based on importance and user preferences.
	•	Applies a sorting mechanism to present content according to behavioral and preference data collected from the user.
	4.	Agent4 - Briefing Creation and Linkage
	•	Generates a briefing report by linking gathered information.
	•	Includes a summary of prioritized content, highlighting key insights from each source link.
	5.	Agent5 - Voice and Video Production
	•	Creates synchronized multimedia content (voice and video).
	•	Allows sharing across platforms to improve accessibility and reach.
	6.	Agent6 - Automation Control
	•	Manages the periodic execution of tasks, running the entire pipeline on a daily schedule to keep content updated.

## Workflow

	1.	Data Collection
	•	The system starts with Agent1, which generates or retrieves a parser for each source link.
	•	Agent2 expands each link, gathering related information for more context.
	2.	Data Prioritization
	•	Agent3 organizes the collected data, prioritizing based on relevance and user preferences.
	3.	Briefing Generation
	•	Agent4 compiles a summary briefing, creating a report that includes structured links and a synthesized overview of the findings.
	4.	Multimedia Production
	•	Agent5 converts the textual briefing into multimedia formats, supporting both voice and video outputs.
	5.	Automation Execution
	•	Agent6 schedules and controls the daily execution, ensuring all agents work in sequence to produce up-to-date output for the user.

## Output

The system generates a comprehensive daily report, outputting it to the user in both text and multimedia formats. The final product can be shared directly across digital platforms.

## Usage

	1.	Configure the links to be processed and set user preferences.
	2.	Run Agent6 to start the daily automation cycle.
	3.	Retrieve generated reports and multimedia content from the output/ directory.

## Future Improvements

	•	Enhancing the prioritization algorithm in Agent3 for better user-specific recommendations.
	•	Expanding Agent5 capabilities to include more multimedia formats.
	•	Adding a real-time mode in Agent6 for on-demand data processing.
