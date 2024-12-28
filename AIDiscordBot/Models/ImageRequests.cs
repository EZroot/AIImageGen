using System.Text.Json.Serialization;

namespace DiscordMusicBot.Services.Models
{
    // Define the request payload structure
    public class GenerateImageRequest
    {
        [JsonPropertyName("prompt")]
        public string Prompt { get; set; }

        [JsonPropertyName("negative_prompt")]
        public string Negative_Prompt { get; set; } = "";

        [JsonPropertyName("num_inference_steps")]
        public int Num_Inference_Steps { get; set; } = 50;

        [JsonPropertyName("width")]
        public int Width { get; set; } = 512;

        [JsonPropertyName("height")]
        public int Height { get; set; } = 512;
    }

    // Define the response structure
    public class GenerateImageResponse
    {
        [JsonPropertyName("message")]
        public string Message { get; set; }

        [JsonPropertyName("image_url")]
        public string Image_Url { get; set; }
        
        [JsonPropertyName("file_path")]
        public string File_Path { get; set; }

        [JsonPropertyName("prompt")]
        public string Prompt { get; set; }

        [JsonPropertyName("negative_prompt")]
        public string Negative_Prompt { get; set; }

        [JsonPropertyName("error")]
        public string Error { get; set; }
    }
}
