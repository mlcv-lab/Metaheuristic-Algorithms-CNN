function net = SAoptimasiCNNmay2015(net, y)
   
    
    %-----------------------------%
    %-----Simulated Annealing-----%
    %-----------------------------%
    iniT=1;          %Temperatur maksimum
    k=1;                     %Konstanta boltzman
    c=0.5;                   %faktor reduksi
    
    it=1;
    deltaf=1;
while abs(deltaf)>1e-8
        while it<10   
          
            net.newffW = net.ffW + 0.00015 * randn(10,192); 
            net.newffb = net.ffb + 0.00015 * randn(10,1);   
           
            %Hitung lost function
            net.newo = sigm(net.newffW * net.fv + repmat(net.newffb, 1, size(net.fv, 2)));
            net.newe = net.newo - y;
            net.newL = 1/2* sum(net.newe(:) .^ 2) / size(net.newe, 2);
            deltaf=net.newL-net.L;
            
                if deltaf<0
                    net.ffW=net.newffW;
                    net.ffb=net.newffb;
                    net.L=net.newL;
                else                    
                    if (iniT*(1+rand*(3-1)))>exp(-deltaf/k)
                        net.ffW=net.newffW;
                        net.ffb=net.newffb;
                        net.L=net.newL;
                    end
                end      
                it=it+1;
        end
            iniT=c*iniT;
            it=1;
            if iniT<1e-8
                break
            end
  
end
