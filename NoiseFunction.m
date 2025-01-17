function noise = NoiseFunction(r, f, wavelength)
    %(a,b,c,d,g) for 690, (a,b,c,d,g) for 850
    nc = [0.2502, 0.02913, 4.625e-6, 0.2128, 6.769e-4; 0.6019, 0.01052, 9.685e-5, 0.1382, 6.785e-4]';
    ws = 0; % wavelength selector
    if wavelength == 750
        ws = 1;
    elseif wavelength == 850
        ws = 2;
    else
        error('invalid wavelength for noise model')
    end
    r(r>50) = 0;
    noise = (nc(1, ws)*exp(nc(2,ws)*r) + nc(3,ws)*exp(nc(4,ws)*r)) * 10^(nc(5, ws)*(f-140));
end


function NoiseTest()
    for wavelength=750:100:850
        figure;
        for freq=0:100:1000
            plot(10:50,NoiseFunction(10:50,freq,wavelength))
            hold on;
        end
        hold off;
    end
end