.PONY: awslab
awslab:
	@echo "Creating virtual environment..."
	@conda env create --name awslab --file diffusion/diffusers.yml
	@echo "Virtual environment created."

.PONY: diffusers
diffusers:
	@echo "Creating virtual environment..."
	@conda env create --name diffusers --file diffusers.yml
	@echo "Virtual environment created."

.PONY: newdiffusers
newdiffusers:
	@echo "Creating virtual environment with python 3.11..."
	@conda create --name diffusers python=3.11 -y
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


###################################################################################################


.PONY: newimp
newimp:
	@echo "Creating virtual environment with python 3.11..."
	@conda create --name imp python=3.11 -y
	@echo "Virtual environment created."

.PONY: freezeimp
freezeimp:
	@echo "Freezing dependencies..."
	@conda env export > implementation/implementaion.yml
	@echo "Dependencies frozen."

.PONY: installimp
installimp:
	@echo "Installing dependencies..."
	@conda env create --name imp --file implementation/implementaion.yml
	@echo "Dependencies installed."

###################################################################################################
# Commit and recommit changes to github
.PONY: commit
commit:
	@echo "Committing changes..."
	@git add .
	@git commit -s -m"${message}"
	@git push origin main
	@git log -1
	@echo "Changes committed and pushed to github."

.PONY: recommit
recommit:
	@echo "Committing changes..."
	@git add .
	@git commit -s --amend --no-edit
	@git push -f origin main
	@git log -1
	@echo "Changes committed and pushed to github."