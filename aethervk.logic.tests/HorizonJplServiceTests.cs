using System;
using System.Linq;
using System.Reflection;
using Xunit;
using Moq;
using AetherVk.Logic.Services;
using AetherVk.Logic.Models;

namespace aethervk.logic.tests
{
    public class HorizonJplServiceTests
    {
        private HorizonJplService CreateService()
        {
            return new HorizonJplService(null!, null!, null!);
        }

        private void InvokeParseObjectDataText(HorizonJplService service, string text)
        {
            var method = typeof(HorizonJplService).GetMethod("ParseObjectDataText", BindingFlags.NonPublic | BindingFlags.Instance);
            method!.Invoke(service, new object[] { text });
        }

        [Fact]
        public void ParseObjectDataText_ShouldExtractConstantsAndIgnoreGarbage()
        {
            var service = CreateService();
            string mockResponse = @"
*******************************************************************************
 Revised: Jul 31, 2013             Mars (body 499)                         499

 PHYSICAL PROPERTIES:
  Vol. Mean Radius (km)    =  3389.50     Mass x10^23 (kg)      =   6.4171
  Density (g/cm^3)         =     3.9335   Equat. radius (km)    =  3396.19
*******************************************************************************


*******************************************************************************
Ephemeris / WWW_USER Wed May 26 21:00:00 2026 Pasadena, USA      / Horizons
*******************************************************************************
Target body name: Mars (499)                      {source: mar097}
$$SOE
Some random ephemeris garbage here
$$EOE
";

            InvokeParseObjectDataText(service, mockResponse);

            var objectData = service.ObjectData;
            
            foreach(var item in objectData)
            {
                Console.WriteLine($"PROP: '{item.Property}' VAL: '{item.Value}'");
            }

            // Should have parsed properties (keys might be mangled by the simplistic split, but values should be there)
            var hasRadius = objectData.Any(x => x.Value != null && x.Value.Contains("3389.50"));
            Assert.True(hasRadius, "Radius value not found");

            var hasMass = objectData.Any(x => x.Value != null && x.Value.Contains("6.4171"));
            Assert.True(hasMass, "Mass value not found");

            // Should NOT have garbage from the Ephemeris section or footers
            var hasGarbage = objectData.Any(x => (x.Property != null && x.Property.Contains("Ephemeris")) || 
                                                 (x.Value != null && x.Value.Contains("Ephemeris")));
            Assert.False(hasGarbage, "Garbage from remaining text was included in the data view.");
            
            var hasSoe = objectData.Any(x => (x.Property != null && x.Property.Contains("$$SOE")) || 
                                             (x.Value != null && x.Value.Contains("$$SOE")));
            Assert.False(hasSoe, "$$SOE marker should not be included.");
        }

        [Fact]
        public void ParseObjectDataText_NoPhysicalPropertiesHeader_ShouldFallbackCleanly()
        {
            var service = CreateService();
            string mockResponse = @"
*******************************************************************************
Some target with no properties block
---
$$SOE
$$EOE
";
            InvokeParseObjectDataText(service, mockResponse);
            
            var objectData = service.ObjectData;
            var hasGarbage = objectData.Any(x => x.Property.Contains("$$SOE") || (x.Value != null && x.Value.Contains("$$SOE")));
            Assert.False(hasGarbage, "Garbage like $$SOE was included when fallback was triggered.");
        }
    }
}
