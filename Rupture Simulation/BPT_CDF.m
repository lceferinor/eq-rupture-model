%% Function: Brownian Time Passage
function CDF = BPT_CDF(points, a, u)
    CDF = [];
    for x = points
        if x <= 0
            cdf = 0;
        else
            cdf = normcdf((sqrt(x/u)-sqrt(u/x))/a) + ...
                exp(2/a^2)*normcdf(-(sqrt(x/u)+sqrt(u/x))/a); 
        end
        CDF = [CDF, cdf];
    end
end