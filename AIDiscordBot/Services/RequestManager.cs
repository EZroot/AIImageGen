using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using DiscordMusicBot.Services.Interfaces;
using DiscordMusicBot.Services.Models;

namespace DiscordMusicBot.Services.Services
{
    internal class RequestManager : IServiceRequestManager
    {
        private HttpClient _client;
        private string _serverBaseUrl;

        public async Task Initialize()
        {
            _client = new HttpClient
            {
                Timeout = TimeSpan.FromMinutes(5) // Set to desired timeout duration
            };
            
            _serverBaseUrl = "http://127.0.0.1:5000"; // Update if different
            await Task.CompletedTask;
        }

        public async Task<string> SendRequestAsync(string prompt, int width = 768, int height = 1024, int inferenceSteps = 50,
                                                    string helperPrompt = "highly detailed, absurdres, highly-detailed, best quality, masterpiece, very aesthetic, ",
                                                    string negativePrompt = "lowres, worst quality, low quality, bad hands,")
        {
            Utils.Debug.Log($"Waiting on image... (~30 secs)");
                        // Configure the request
            var requestPayload = new GenerateImageRequest
            {
                Prompt = helperPrompt+prompt,
                Negative_Prompt = negativePrompt,
                Num_Inference_Steps = inferenceSteps,
                Width = width,
                Height = height
            };

            // Serialize the request payload to JSON
            var json = JsonSerializer.Serialize(requestPayload);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            try
            {
                // Send POST request to the /generate endpoint
                var response = await _client.PostAsync($"{_serverBaseUrl}/generate", content);

                // Ensure the request was successful
                response.EnsureSuccessStatusCode();

                // Read and deserialize the response JSON
                var responseString = await response.Content.ReadAsStringAsync();
                var options = new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                };
                var generateResponse = JsonSerializer.Deserialize<GenerateImageResponse>(responseString, options);

                if (!string.IsNullOrEmpty(generateResponse.Error))
                {
                    Utils.Debug.Log($"Error: {generateResponse.Error}");
                    return null;
                }
                else
                {
                    Utils.Debug.Log($"Image generated successfully!");
                    // Utils.Debug.Log($"Prompt: {generateResponse.Prompt}");
                    // Utils.Debug.Log($"Image URL: {generateResponse.Image_Url}");
                    // Utils.Debug.Log($"File Path: {generateResponse.File_Path}");
                    // Extract the filename from the image URL
                    var filePath = generateResponse.File_Path;

                    if (File.Exists(filePath))
                    {
                        Utils.Debug.Log($"<color=green>Accessing image directly from:</color> <color=cyan>{filePath}");
                        return filePath;
                    }
                    else
                    {
                        Utils.Debug.Log($"<color=red>Image file not found at: {filePath}");
                        return null;
                    }
                }
            }
            catch (HttpRequestException e)
            {
                Utils.Debug.Log($"<color=red>Request error: {e.Message}");
                return e.Message;
            }
            catch (JsonException e)
            {
                Utils.Debug.Log($"<color=red>JSON parsing error: {e.Message}");
                return e.Message;
            }
            catch (Exception e)
            {
                Utils.Debug.Log($"<color=red>Unexpected error: {e.Message}");
                return e.Message;
            }
        }
    }
}
