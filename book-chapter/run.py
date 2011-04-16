
from dolfin_utils import pjobs


jobs = ["sh poisson_neumann.sh > poisson_neumann.output", "sh stokes.sh > stokes.output", "sh timestokes.sh > timestokes.output", "sh hodge.sh > hodge.output" ] 

pjobs.submit(jobs)
