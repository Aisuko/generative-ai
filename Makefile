.PONY: diffusers
diffusers:
	@echo "Creating virtual environment..."
	@conda env create -f diffusers.yml
	@echo "Virtual environment created."

.PONY: newenv
newenv:
	@echo "Creating virtual environment with python 3.11..."
	@conda create -n diffusers python=3.11 -y
	@echo "Virtual environment created."

.PONY: torch
torch:
	@echo "Installing torch..."
	@conda install pytorch::pytorch torchvision torchaudio -c pytorch -y
	@echo "Torch installed."

.PONY: install
install:
	@echo "Installing dependencies..."
	@pip install diffusers["torch","flax"] transformers
	@echo "Dependencies installed."

.PONY: freeze
freeze:
	@echo "Freezing dependencies..."
	@conda env export > diffusers.yml
	@echo "Dependencies frozen."