try:
	from dotenv import load_dotenv
	load_dotenv()
except Exception:
	# If python-dotenv is not available, fall back to existing environment variables
	pass
