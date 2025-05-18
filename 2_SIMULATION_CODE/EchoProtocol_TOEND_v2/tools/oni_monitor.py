# tools/oni_monitor.py  
import time  
from core import EntropicIdentity  
from monitor import FractonLogger  

class ONIMonitor:  
    def __init__(self, format='text'):  
        self.identity = EntropicIdentity()  
        self.logger = FractonLogger()  
        self.format = format  

    def _format_output(self):  
        data = {  
            "mu": self.identity.μ,  
            "sigma": self.identity.σ,  
            "lambda": self.identity.λ,  
            "phase": self.identity.get_phase().name  
        }  
        if self.format == 'json':  
            return json.dumps(data, indent=2)  
        elif self.format == 'md':  
            return f"| {data['mu']:.2f} | {data['sigma']:.2f} | {data['lambda']:.2f} | {data['phase']} |"  
        else:  # Mode texte  
            return f"μ={data['mu']:.2f} σ={data['sigma']:.2f} λ={data['lambda']:.2f} | Phase: {data['phase']}"  

    def stream(self, interval=1.0):  
        try:  
            while True:  
                print(self._format_output())  
                time.sleep(interval)  
        except KeyboardInterrupt:  
            print("\n🌀 Monitoring stopped.")  